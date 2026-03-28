// Creates federated identity credentials on the Secrets Store CSI Driver addon
// identity so that the CSI Driver can authenticate to Azure Key Vault on behalf
// of pods that mount TLS certificates via a SecretProviderClass.
//
// Background
// ----------
// When a SecretProviderClass specifies a clientID the CSI Driver uses
// Workload Identity federation:
//   1. The Kubernetes API server projects a signed service account token into
//      each pod that mounts the CSI volume.
//   2. The CSI Driver exchanges that token for an Azure AD token by calling the
//      OIDC /token endpoint and presenting the pod SA token as a federated
//      assertion for the identity given by clientID.
//   3. Azure AD validates the assertion by looking up the federated credential
//      whose issuer matches the AKS OIDC issuer URL and whose subject matches
//      the pod's service account ("system:serviceaccount:<ns>:<sa>").
//
// A federated credential must therefore exist for every Kubernetes service
// account that is used by pods mounting the CSI TLS volume.
//
// This module must be deployed scoped to the MC_ managed resource group where
// AKS places the CSI addon identity, and must run after the AKS cluster exists.
//
// Parameters
// ----------
// clusterName  – AKS cluster name; used to derive the addon identity name.
// oidcIssuerUrl – OIDC issuer URL of the cluster
//                 (aksCluster.properties.oidcIssuerProfile.issuerURL).

@description('Name of the AKS cluster (used to locate the CSI addon identity in the MC_ resource group).')
param clusterName string

@description('OIDC issuer URL of the AKS cluster (from aksCluster.properties.oidcIssuerProfile.issuerURL).')
param oidcIssuerUrl string

// The CSI addon identity is automatically named 'azurekeyvaultsecretsprovider-<clusterName>'
// by AKS and lives in the MC_ resource group (the scope of this module).
resource csiAddonIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: 'azurekeyvaultsecretsprovider-${clusterName}'
}

// Federated credential for the StreamWise pod service account.
// Allows the CSI Driver to authenticate to Key Vault on behalf of the
// streamwise pod (serviceAccountName: streamwise-service-account).
resource streamwiseFederatedCred 'Microsoft.ManagedIdentity/userAssignedIdentities/federatedIdentityCredentials@2023-01-31' = {
  parent: csiAddonIdentity
  name: 'streamwise-service-account'
  properties: {
    issuer: oidcIssuerUrl
    subject: 'system:serviceaccount:rtgen:streamwise-service-account'
    audiences: ['api://AzureADTokenExchange']
  }
}

// Federated credential for the StreamCast / StreamWise-app pod service account.
// Allows the CSI Driver to authenticate to Key Vault on behalf of the
// streamcast pod (serviceAccountName: streamwiseapp-service-account).
resource streamwiseAppFederatedCred 'Microsoft.ManagedIdentity/userAssignedIdentities/federatedIdentityCredentials@2023-01-31' = {
  parent: csiAddonIdentity
  name: 'streamwiseapp-service-account'
  properties: {
    issuer: oidcIssuerUrl
    subject: 'system:serviceaccount:rtgen:streamwiseapp-service-account'
    audiences: ['api://AzureADTokenExchange']
  }
}

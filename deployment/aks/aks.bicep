@description('Name of the AKS cluster. Defaults to the resource group name suffixed with -cluster.')
param clusterName string = '${resourceGroup().name}-cluster'

@description('Location for the AKS cluster. Defaults to the resource group location.')
param location string = resourceGroup().location

@description('VM size for the system node pool.')
param systemNodeVmSize string = 'Standard_D16s_v5'

@description('Number of nodes in the system node pool.')
param systemNodeCount int = 1

// Some options:
// Standard_NC96ads_A100_v4
// Standard_ND96ams_A100_v4
// Standard_ND96isrf_H100_v5
// Standard_ND96isr_H200_v5
// Standard_ND128isr_NDR_GB200_v6
@description('VM size for the GPU spot node pool.')
param gpuNodeVmSize string = 'Standard_ND96ams_A100_v4'

@description('Initial number of nodes in the GPU spot node pool (full GPUs, no MIG).')
param gpuNodeCount int = 0

@description('Name of the GPU spot node pool (full GPUs).')
param gpuNodePoolName string = 'spoth100'

@description('VM size for the GPU MIG spot node pool. Defaults to the same size as the full-GPU pool.')
param gpuMigNodeVmSize string = gpuNodeVmSize

@description('Initial number of nodes in the GPU MIG spot node pool.')
param gpuMigNodeCount int = 0

@description('Name of the GPU MIG spot node pool.')
param gpuMigNodePoolName string = 'spoth100mig'

@description('Name of an existing ACR to attach to the AKS cluster. Leave empty to skip.')
param acrName string = ''

@description('Resource group of the ACR. Defaults to the AKS resource group.')
param acrResourceGroup string = resourceGroup().name

@description('DNS prefix for the AKS cluster. Defaults to the cluster name.')
param dnsPrefix string = clusterName

// [S360 - SFI-NS2.6.1] Disable default outbound access for all node VMs.
// When true a NAT gateway and dedicated VNet are provisioned for controlled
// outbound connectivity instead of unrestricted default access.
// Set to false to opt out (e.g. for quick testing).
param disableDefaultOutboundAccess bool = true

@description('Address prefix for the AKS VNet. Only used when disableDefaultOutboundAccess is true.')
// Uses 10.10.0.0/16 to avoid overlap with the VM-based K8s deployment (vm-gpu-k8s-deployment.bicep)
// which occupies 10.0.0.0/16 – 10.5.0.0/16.  Choose a different block if those ranges are taken.
param vnetAddressPrefix string = '10.10.0.0/16'

@description('Address prefix for the AKS node subnet. Only used when disableDefaultOutboundAccess is true.')
param subnetAddressPrefix string = '10.10.0.0/24'

// Key Vault name must be globally unique, 3-24 chars, alphanumeric + hyphens, start with letter.
// The default uses a stable hash of the resource group ID so repeated deployments reuse the same vault.
@description('Name of the Azure Key Vault for TLS certificates. Must be globally unique (3–24 chars).')
@maxLength(24)
param keyVaultName string = 'kv-${take(uniqueString(resourceGroup().id), 20)}'

@description('Name of the TLS certificate stored in Key Vault (used as a fallback or for manual cert import).')
param tlsCertificateName string = 'streamwise-tls'

// When true, provisions Azure Key Vault, a self-signed TLS certificate (as a
// bootstrap fallback), enables the Secrets Store CSI Driver addon with OIDC
// issuer + workload identity, opens port 80 for ACME HTTP-01 challenges, and
// grants the CSI addon identity read access to Key Vault.
// For browser-trusted HTTPS, run deployment/aks/setup-letsencrypt.sh after
// cluster deployment to replace the self-signed cert with a CA-signed
// Let's Encrypt certificate (or set LETSENCRYPT_EMAIL in quick-deploy.sh).
param enableSecureSetup bool = false

@description('''DNS label prefix applied to the pods public IP address.
The resulting FQDN (<dnsLabelPrefix>.<region>.cloudapp.azure.com) is required for
cert-manager / Let\'s Encrypt CA-signed certificates.
Defaults to a unique, stable label derived from the resource-group ID.
Set to an empty string to omit the DNS label (self-signed cert only).''')
param dnsLabelPrefix string = 'streamwise-${take(uniqueString(resourceGroup().id), 6)}'


// ---------------------------------------------------------------------------
// Public IP – used by Kubernetes LoadBalancer services (StreamWise, StreamCast)
// A DNS label is attached when dnsLabelPrefix is non-empty, which provides the
// FQDN required by cert-manager / Let's Encrypt for CA-signed TLS certificates.
// ---------------------------------------------------------------------------
resource publicIp 'Microsoft.Network/publicIPAddresses@2023-11-01' = {
  name: 'aks-pods-public-ip'
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIPAllocationMethod: 'Static'
    // Setting dnsSettings to null (when dnsLabelPrefix is empty) is the standard
    // Bicep pattern for conditional properties — ARM treats null as property omission.
    dnsSettings: !empty(dnsLabelPrefix) ? {
      domainNameLabel: dnsLabelPrefix
    } : null
    ipTags: [
      {
        ipTagType: 'FirstPartyUsage'
        tag: '/NonProd'
      }
    ]
  }
}


// ---------------------------------------------------------------------------
// NAT Gateway resources – only provisioned when disableDefaultOutboundAccess = true
// ---------------------------------------------------------------------------
resource natPublicIp 'Microsoft.Network/publicIPAddresses@2023-11-01' = if (disableDefaultOutboundAccess) {
  name: 'aks-nat-public-ip'
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIPAllocationMethod: 'Static'
    ipTags: [
      {
        ipTagType: 'FirstPartyUsage'
        tag: '/NonProd'
      }
    ]
  }
}

resource natGateway 'Microsoft.Network/natGateways@2023-11-01' = if (disableDefaultOutboundAccess) {
  name: 'aks-nat-gateway'
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIpAddresses: [
      {
        id: natPublicIp.id
      }
    ]
    idleTimeoutInMinutes: 4
  }
}

// ---------------------------------------------------------------------------
// Network Security Group – allows inbound traffic on ports 8000–9000 so that
// Kubernetes LoadBalancer services (StreamWise, StreamCast, model wrappers)
// are reachable from the Internet.  Without this explicit NSG, corporate
// policy may attach a default-deny NSG to the subnet that blocks traffic
// even when the NIC-level NSG (managed by AKS) allows it.
// Port 80 is also opened to allow the cert-manager ACME HTTP-01 solver to
// respond to Let's Encrypt validation challenges for CA-signed certificates.
// ---------------------------------------------------------------------------
resource aksNsg 'Microsoft.Network/networkSecurityGroups@2023-11-01' = if (disableDefaultOutboundAccess) {
  name: 'aks-node-subnet-nsg'
  location: location
  properties: {
    securityRules: [
      {
        name: 'AllowServicePortsInbound'
        properties: {
          priority: 100
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourcePortRange: '*'
          sourceAddressPrefix: 'Internet'
          destinationAddressPrefix: publicIp.properties.ipAddress
          destinationPortRange: '8000-9000'
        }
      }
      {
        name: 'AllowAcmeHttp01Inbound'
        properties: {
          priority: 110
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourcePortRange: '*'
          sourceAddressPrefix: 'Internet'
          destinationAddressPrefix: publicIp.properties.ipAddress
          destinationPortRange: '80'
        }
      }
    ]
  }
}

// ---------------------------------------------------------------------------
// Custom VNet – only provisioned when disableDefaultOutboundAccess = true.
// The node subnet has defaultOutboundAccess disabled and uses the NAT gateway
// for controlled outbound connectivity.  The NSG above is attached to the
// subnet to ensure LoadBalancer service ports are reachable.
// ---------------------------------------------------------------------------
resource aksVnet 'Microsoft.Network/virtualNetworks@2023-11-01' = if (disableDefaultOutboundAccess) {
  name: 'aks-vnet'
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        vnetAddressPrefix
      ]
    }
    subnets: [
      {
        name: 'aks-node-subnet'
        properties: {
          addressPrefix: subnetAddressPrefix
          // [S360 - SFI-NS2.6.1] disable default outbound access for all subnets
          defaultOutboundAccess: false
          natGateway: {
            id: natGateway.id
          }
          networkSecurityGroup: {
            id: aksNsg.id
          }
        }
      }
    ]
  }
}


// ---------------------------------------------------------------------------
// Subnet ID used by both node pools when disableDefaultOutboundAccess is true.
// Evaluates to null (property omitted) when the custom VNet is not provisioned.
// ---------------------------------------------------------------------------
var nodeSubnetId = disableDefaultOutboundAccess ? any(aksVnet).properties.subnets[0].id : null

// When disableDefaultOutboundAccess is false, networkProfile is null so it is
// omitted from the ARM template and AKS uses its default settings
// (kubenet plugin, loadBalancer outbound type).
var networkProfile = disableDefaultOutboundAccess ? {
  outboundType: 'userAssignedNATGateway'
} : null


// ---------------------------------------------------------------------------
// Azure Key Vault – stores the TLS certificate used by StreamWise and apps.
// RBAC authorization is used so that the AKS kubelet identity can read
// certificates and secrets without a legacy access policy.
// Only provisioned when enableSecureSetup is true.
// ---------------------------------------------------------------------------
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = if (enableSecureSetup) {
  name: keyVaultName
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

// Fallback self-signed TLS certificate stored in Key Vault.
// This bootstraps HTTPS immediately so pods can start while cert-manager
// provisions the CA-signed Let's Encrypt certificate (1–3 minutes).
// To replace with a CA-signed certificate:
//   a) Run deployment/aks/setup-letsencrypt.sh (automated, recommended), or
//   b) Follow the manual guide in deployment/k8s/certs.md, or
//   c) Import a PEM/PFX file: az keyvault certificate import ...
// Only provisioned when enableSecureSetup is true.
#disable-next-line BCP081
resource tlsCertificate 'Microsoft.KeyVault/vaults/certificates@2023-07-01' = if (enableSecureSetup) {
  parent: keyVault
  name: tlsCertificateName
  properties: {
    certificatePolicy: {
      keyProperties: {
        exportable: true
        keyType: 'RSA'
        keySize: 4096
        reuseKey: false
      }
      secretProperties: {
        // PEM encoding: the corresponding secret contains both the private key
        // and the certificate as concatenated PEM blocks.
        contentType: 'application/x-pem-file'
      }
      x509CertificateProperties: {
        subject: 'CN=streamwise'
        validityInMonths: 12
        keyUsage: [
          'digitalSignature'
          'keyEncipherment'
        ]
        // TLS server authentication EKU
        ekus: [
          '1.3.6.1.5.5.7.3.1'
        ]
      }
      issuerParameters: {
        // Self-signed fallback; the cert-manager guide in deployment/k8s/certs.md
        // shows how to replace this with a CA-signed certificate.
        name: 'Self'
      }
      lifetimeActions: [
        {
          trigger: {
            daysBeforeExpiry: 30
          }
          action: {
            actionType: 'AutoRenew'
          }
        }
      ]
    }
  }
}


// ---------------------------------------------------------------------------
// AKS Cluster
// ---------------------------------------------------------------------------
resource aksCluster 'Microsoft.ContainerService/managedClusters@2024-09-01' = {
  name: clusterName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    dnsPrefix: dnsPrefix
    agentPoolProfiles: [
      {
        name: 'nodepool1'
        count: systemNodeCount
        vmSize: systemNodeVmSize
        osType: 'Linux'
        mode: 'System'
        vnetSubnetID: nodeSubnetId
      }
    ]
    networkProfile: networkProfile
    // Enable the Secrets Store CSI Driver with Azure Key Vault provider so
    // that pods can mount Key Vault certificates directly as volume files.
    // Only enabled when enableSecureSetup is true.
    addonProfiles: enableSecureSetup ? {
      azureKeyvaultSecretsProvider: {
        enabled: true
        config: {
          enableSecretRotation: 'true'
          rotationPollInterval: '2m'
        }
      }
    } : {}
    // Enable OIDC issuer and workload identity so that the Secrets Store CSI
    // Driver can authenticate to Azure Key Vault using the addon's managed
    // identity via the clientID field in SecretProviderClass.
    // Only enabled when enableSecureSetup is true.
    oidcIssuerProfile: enableSecureSetup ? {
      enabled: true
    } : null
    securityProfile: enableSecureSetup ? {
      workloadIdentity: {
        enabled: true
      }
    } : null
  }
}


// ---------------------------------------------------------------------------
// GPU Spot Node Pool – full GPUs (no MIG)
// ---------------------------------------------------------------------------
resource gpuNodePool 'Microsoft.ContainerService/managedClusters/agentPools@2024-09-01' = {
  parent: aksCluster
  name: gpuNodePoolName
  properties: {
    count: gpuNodeCount
    vmSize: gpuNodeVmSize
    osType: 'Linux'
    mode: 'User'
    scaleSetPriority: 'Spot'
    scaleSetEvictionPolicy: 'Delete'
    spotMaxPrice: -1
    nodeTaints: [
      'kubernetes.azure.com/scalesetpriority=spot:NoSchedule'
    ]
    nodeLabels: {
      'kubernetes.azure.com/scalesetpriority': 'spot'
    }
    vnetSubnetID: nodeSubnetId
  }
}

// ---------------------------------------------------------------------------
// GPU MIG Spot Node Pool – nodes where MIG is manually configured
// (e.g. 7 full GPUs + 1 MIG-partitioned GPU for lightweight services)
// ---------------------------------------------------------------------------
resource gpuMigNodePool 'Microsoft.ContainerService/managedClusters/agentPools@2024-09-01' = {
  parent: aksCluster
  name: gpuMigNodePoolName
  properties: {
    count: gpuMigNodeCount
    vmSize: gpuMigNodeVmSize
    osType: 'Linux'
    mode: 'User'
    scaleSetPriority: 'Spot'
    scaleSetEvictionPolicy: 'Delete'
    spotMaxPrice: -1
    nodeTaints: [
      'kubernetes.azure.com/scalesetpriority=spot:NoSchedule'
    ]
    nodeLabels: {
      'kubernetes.azure.com/scalesetpriority': 'spot'
      'gpu-config': 'mig'
    }
    vnetSubnetID: nodeSubnetId
  }
}

// ---------------------------------------------------------------------------
// ACR Role Assignment
// ---------------------------------------------------------------------------
// When the ACR is in a different resource group, use a module for cross-scope
// deployment (see ../bicep/roleACRAssignment.bicep for an example).
// For same-resource-group ACR, or if you attached the ACR via CLI:
//   az aks update -g <rg> --name <cluster> --attach-acr <acrName>
// you can skip this section.

module acrRoleAssignment '../bicep/roleACRAssignment.bicep' = if (!empty(acrName)) {
  name: 'acr-role-assignment'
  scope: resourceGroup(acrResourceGroup)
  params: {
    acrName: acrName
    principalId: aksCluster.properties.identityProfile.kubeletidentity.objectId
  }
}

// ---------------------------------------------------------------------------
// Network Contributor Role – allows AKS to manage the public IP for
// LoadBalancer services when the IP is in the same resource group as the
// cluster (not the MC_ node resource group).
// ---------------------------------------------------------------------------
var networkContributorRoleId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  '4d97b98b-1d4f-4787-a291-c67834d212e7'
)

resource networkContributorAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(aksCluster.id, networkContributorRoleId, resourceGroup().id)
  properties: {
    principalId: aksCluster.identity.principalId
    roleDefinitionId: networkContributorRoleId
    principalType: 'ServicePrincipal'
  }
}

// ---------------------------------------------------------------------------
// Key Vault RBAC – grant the Secrets Store CSI Driver addon identity read
// access to Key Vault secrets and certificates so it can retrieve the TLS
// certificate at pod startup via the clientID in SecretProviderClass.
//
// The addon creates its own user-assigned managed identity
// (addonProfiles.azureKeyvaultSecretsProvider.identity).  Using the addon
// identity (not the kubelet identity) is required when the SecretProviderClass
// specifies a clientID and workload identity is enabled.
//
//   Key Vault Secrets User  (4633458b-…) – read secrets (PEM bundle + key)
//   Key Vault Certificate User (db79e9a7-…) – read certificate public part
//
// Only provisioned when enableSecureSetup is true.
// ---------------------------------------------------------------------------
var kvSecretsUserRoleId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  '4633458b-17de-408a-b874-0445c86b69e6'
)

var kvCertificateUserRoleId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  'db79e9a7-68ee-4b58-9aeb-b90e7c24fcba'
)

resource kvSecretsUserAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (enableSecureSetup) {
  scope: keyVault
  name: guid(keyVault.id, aksCluster.id, kvSecretsUserRoleId)
  properties: {
    principalId: aksCluster.properties.addonProfiles.azureKeyvaultSecretsProvider.identity.objectId
    roleDefinitionId: kvSecretsUserRoleId
    principalType: 'ServicePrincipal'
  }
}

resource kvCertificateUserAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (enableSecureSetup) {
  scope: keyVault
  name: guid(keyVault.id, aksCluster.id, kvCertificateUserRoleId)
  properties: {
    principalId: aksCluster.properties.addonProfiles.azureKeyvaultSecretsProvider.identity.objectId
    roleDefinitionId: kvCertificateUserRoleId
    principalType: 'ServicePrincipal'
  }
}

// ---------------------------------------------------------------------------
// Federated Identity Credentials – required for workload identity exchange.
//
// When the CSI Driver mounts a Key Vault secret using clientID + workload
// identity, it presents the pod's projected service account token as a
// federated assertion to Azure AD.  Azure AD validates the assertion against
// a federated credential registered on the CSI addon identity that matches:
//   issuer  = the cluster OIDC issuer URL
//   subject = system:serviceaccount:<namespace>:<service-account-name>
//
// One federated credential is required per Kubernetes service account whose
// pod mounts the TLS CSI volume.  The addon identity and its federated
// credentials live in the MC_ managed resource group created by AKS.
// The module is scoped to that group and depends on the AKS cluster so
// it runs after the cluster (and the CSI addon identity) is provisioned.
// ---------------------------------------------------------------------------
var mcResourceGroup = 'MC_${resourceGroup().name}_${clusterName}_${location}'

module csiAddonFederatedCreds '../bicep/csi-federated-credentials.bicep' = if (enableSecureSetup) {
  name: 'csi-addon-federated-creds'
  scope: resourceGroup(mcResourceGroup)
  params: {
    clusterName: clusterName
    // Guard with the same flag so the expression is only evaluated when
    // oidcIssuerProfile is actually enabled on the cluster.
    oidcIssuerUrl: enableSecureSetup ? aksCluster.properties.oidcIssuerProfile.issuerURL : ''
  }
}


// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

// Guard the CSI addon client ID behind the same flag to avoid evaluating the
// nested property chain when azureKeyvaultSecretsProvider is not in addonProfiles.
var csiAddonClientIdValue = enableSecureSetup ? aksCluster.properties.addonProfiles.azureKeyvaultSecretsProvider.identity.clientId : ''

output clusterName string = aksCluster.name
output publicIpAddress string = publicIp.properties.ipAddress
output publicIpName string = publicIp.name
output publicFqdn string = !empty(dnsLabelPrefix) ? publicIp.properties.dnsSettings.fqdn : ''
output gpuNodePoolName string = gpuNodePool.name
output gpuMigNodePoolName string = gpuMigNodePool.name
output keyVaultName string = enableSecureSetup ? keyVault.name : ''
output tlsCertificateName string = enableSecureSetup ? tlsCertificate.name : ''
output csiAddonClientId string = csiAddonClientIdValue
output tenantId string = subscription().tenantId

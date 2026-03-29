// aks-frontdoor.bicep — AKS cluster + Azure Front Door Premium (managed HTTPS
// via Private Link, bypasses NRMS restrictions).
//
// Architecture:
//   Client  ──HTTPS──▶  Front Door Premium (*.azurefd.net, managed cert)
//                          │
//                          │  Private Link (Azure backbone — no public inbound)
//                          ▼
//                       Private Link Service
//                          │
//                          ▼
//                       AKS Internal Load Balancer (HTTP)
//                          │
//                          ▼
//                       Pods: StreamWise (:18181), StreamCast (:18080)
//
// This Bicep deploys Phase 1 (AKS cluster + networking).
// After deploying pods, run the companion script to create the Private Link
// Service and Front Door Premium (Phase 2):
//
//   bash deployment/aks/setup-frontdoor.sh
//
// Usage:
//   az deployment group create \
//     --resource-group <rg> \
//     --template-file deployment/aks/aks-frontdoor.bicep \
//     --parameters acrName=<acr> acrResourceGroup=<acr-rg>

@description('Name of the AKS cluster. Defaults to the resource group name suffixed with -cluster.')
param clusterName string = '${resourceGroup().name}-cluster'

@description('Location for all resources. Defaults to the resource group location.')
param location string = resourceGroup().location

@description('VM size for the system node pool.')
param systemNodeVmSize string = 'Standard_D16ds_v5'

@description('Number of nodes in the system node pool.')
param systemNodeCount int = 1

@description('Name of an existing ACR to attach to the AKS cluster. Leave empty to skip.')
param acrName string = ''

@description('Resource group of the ACR. Defaults to the AKS resource group.')
param acrResourceGroup string = resourceGroup().name

@description('DNS prefix for the AKS cluster. Defaults to the cluster name.')
param dnsPrefix string = clusterName

// [S360 - SFI-NS2.6.1] Disable default outbound access for all node VMs.
param disableDefaultOutboundAccess bool = true

@description('Address prefix for the AKS VNet.')
param vnetAddressPrefix string = '10.10.0.0/16'

@description('Address prefix for the AKS node subnet.')
param subnetAddressPrefix string = '10.10.0.0/24'

@description('Address prefix for the Private Link Service subnet.')
param plsSubnetAddressPrefix string = '10.10.1.0/24'


// ---------------------------------------------------------------------------
// Public IP – used by Kubernetes LoadBalancer services (StreamWise, StreamCast)
// ---------------------------------------------------------------------------
resource publicIp 'Microsoft.Network/publicIPAddresses@2023-11-01' = {
  name: 'aks-pods-public-ip'
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
// NSG – allows inbound traffic on service ports and K8s NodePorts.
// The AllowK8sNodePorts rule is critical: Azure LB performs DNAT to NodePorts
// (30000–32767) on node private IPs, so matching only the public IP on
// service ports (8000–9000) is insufficient for Internet clients.
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
        name: 'AllowK8sNodePorts'
        properties: {
          priority: 102
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourcePortRange: '*'
          sourceAddressPrefix: 'Internet'
          destinationAddressPrefix: 'VirtualNetwork'
          destinationPortRange: '30000-32767'
        }
      }
      {
        // Azure Front Door health probes and forwarded requests originate from
        // the AzureFrontDoor.Backend service tag.  Allow them to reach the
        // service ports and K8s NodePorts so that Front Door can route traffic.
        name: 'AllowFrontDoorBackend'
        properties: {
          priority: 112
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourcePortRange: '*'
          sourceAddressPrefix: 'AzureFrontDoor.Backend'
          destinationAddressPrefix: 'VirtualNetwork'
          destinationPortRanges: [
            '8080-8081'
            '30000-32767'
          ]
        }
      }
    ]
  }
}


// ---------------------------------------------------------------------------
// VNet – only provisioned when disableDefaultOutboundAccess = true
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
          defaultOutboundAccess: false
          natGateway: {
            id: natGateway.id
          }
          networkSecurityGroup: {
            id: aksNsg.id
          }
        }
      }
      {
        // Dedicated subnet for Private Link Service.
        // privateLinkServiceNetworkPolicies must be Disabled so that
        // PLS resources can be created in this subnet.
        name: 'pls-subnet'
        properties: {
          addressPrefix: plsSubnetAddressPrefix
          privateLinkServiceNetworkPolicies: 'Disabled'
          defaultOutboundAccess: false
          natGateway: {
            id: natGateway.id
          }
        }
      }
    ]
  }
}

var nodeSubnetId = disableDefaultOutboundAccess ? any(aksVnet).properties.subnets[0].id : null
var networkProfile = disableDefaultOutboundAccess ? {
  outboundType: 'userAssignedNATGateway'
} : null


// ---------------------------------------------------------------------------
// AKS Cluster – system node pool only (no GPUs)
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
  }
}


// ---------------------------------------------------------------------------
// ACR Role Assignment
// ---------------------------------------------------------------------------
module acrRoleAssignment '../bicep/roleACRAssignment.bicep' = if (!empty(acrName)) {
  name: 'acr-role-assignment'
  scope: resourceGroup(acrResourceGroup)
  params: {
    acrName: acrName
    principalId: aksCluster.properties.identityProfile.kubeletidentity.objectId
  }
}


// ---------------------------------------------------------------------------
// Network Contributor Role – allows AKS to manage the public IP
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
// Outputs
// ---------------------------------------------------------------------------
output clusterName string = aksCluster.name
output publicIpAddress string = publicIp.properties.ipAddress
output publicIpName string = publicIp.name
output plsSubnetId string = disableDefaultOutboundAccess ? '${aksVnet.id}/subnets/pls-subnet' : ''
output vnetName string = disableDefaultOutboundAccess ? aksVnet.name : ''

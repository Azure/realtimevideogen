// aks-frontdoor.bicep — AKS cluster + Azure Front Door (managed HTTPS).
//
// Deploys a lightweight AKS cluster (system node pool only, no GPUs) behind
// Azure Front Door Standard.  Front Door provides browser-trusted HTTPS via
// its managed certificate on the *.azurefd.net domain — no Let's Encrypt,
// Key Vault, or cert-manager needed.
//
// Architecture:
//   Client  ──HTTPS──▶  Front Door (*.azurefd.net, managed cert)
//                          │
//                          │  HTTP (forwardingProtocol: HttpOnly)
//                          ▼
//                       AKS LoadBalancer (public IP, ports 8080 / 8081)
//                          │
//                          ▼
//                       Pods: StreamWise (:18181), StreamCast (:18080)
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

@description('Name of the Front Door profile.')
param frontDoorName string = 'afd-${take(uniqueString(resourceGroup().id), 10)}'


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
// Azure Front Door Standard – managed HTTPS on *.azurefd.net
//
// Two endpoints (streamwise, streamcast) each route HTTPS to the AKS
// LoadBalancer origin on the appropriate HTTP port.
// ---------------------------------------------------------------------------
resource frontDoor 'Microsoft.Cdn/profiles@2024-09-01' = {
  name: frontDoorName
  location: 'global'
  sku: {
    name: 'Standard_AzureFrontDoor'
  }
}

// -- StreamWise endpoint (port 8081) --
resource streamwiseEndpoint 'Microsoft.Cdn/profiles/afdEndpoints@2024-09-01' = {
  parent: frontDoor
  name: 'streamwise'
  location: 'global'
}

resource streamwiseOriginGroup 'Microsoft.Cdn/profiles/originGroups@2024-09-01' = {
  parent: frontDoor
  name: 'streamwise-origin-group'
  properties: {
    healthProbeSettings: {
      probePath: '/health'
      probeProtocol: 'Http'
      probeIntervalInSeconds: 30
      probeRequestType: 'GET'
    }
    loadBalancingSettings: {
      sampleSize: 4
      successfulSamplesRequired: 3
      additionalLatencyInMilliseconds: 50
    }
  }
}

resource streamwiseOrigin 'Microsoft.Cdn/profiles/originGroups/origins@2024-09-01' = {
  parent: streamwiseOriginGroup
  name: 'aks-streamwise'
  properties: {
    hostName: publicIp.properties.ipAddress
    httpPort: 8081
    httpsPort: 443
    originHostHeader: publicIp.properties.ipAddress
    priority: 1
    weight: 1000
    enforceCertificateNameCheck: false
  }
}

resource streamwiseRoute 'Microsoft.Cdn/profiles/afdEndpoints/routes@2024-09-01' = {
  parent: streamwiseEndpoint
  name: 'streamwise-route'
  properties: {
    originGroup: {
      id: streamwiseOriginGroup.id
    }
    supportedProtocols: [
      'Http'
      'Https'
    ]
    patternsToMatch: [
      '/*'
    ]
    forwardingProtocol: 'HttpOnly'
    httpsRedirect: 'Enabled'
    linkToDefaultDomain: 'Enabled'
  }
  dependsOn: [
    streamwiseOrigin
  ]
}

// -- StreamCast endpoint (port 8080) --
resource streamcastEndpoint 'Microsoft.Cdn/profiles/afdEndpoints@2024-09-01' = {
  parent: frontDoor
  name: 'streamcast'
  location: 'global'
}

resource streamcastOriginGroup 'Microsoft.Cdn/profiles/originGroups@2024-09-01' = {
  parent: frontDoor
  name: 'streamcast-origin-group'
  properties: {
    healthProbeSettings: {
      probePath: '/health'
      probeProtocol: 'Http'
      probeIntervalInSeconds: 30
      probeRequestType: 'GET'
    }
    loadBalancingSettings: {
      sampleSize: 4
      successfulSamplesRequired: 3
      additionalLatencyInMilliseconds: 50
    }
  }
}

resource streamcastOrigin 'Microsoft.Cdn/profiles/originGroups/origins@2024-09-01' = {
  parent: streamcastOriginGroup
  name: 'aks-streamcast'
  properties: {
    hostName: publicIp.properties.ipAddress
    httpPort: 8080
    httpsPort: 443
    originHostHeader: publicIp.properties.ipAddress
    priority: 1
    weight: 1000
    enforceCertificateNameCheck: false
  }
}

resource streamcastRoute 'Microsoft.Cdn/profiles/afdEndpoints/routes@2024-09-01' = {
  parent: streamcastEndpoint
  name: 'streamcast-route'
  properties: {
    originGroup: {
      id: streamcastOriginGroup.id
    }
    supportedProtocols: [
      'Http'
      'Https'
    ]
    patternsToMatch: [
      '/*'
    ]
    forwardingProtocol: 'HttpOnly'
    httpsRedirect: 'Enabled'
    linkToDefaultDomain: 'Enabled'
  }
  dependsOn: [
    streamcastOrigin
  ]
}


// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------
output clusterName string = aksCluster.name
output publicIpAddress string = publicIp.properties.ipAddress
output publicIpName string = publicIp.name
output frontDoorName string = frontDoor.name
output streamwiseUrl string = 'https://${streamwiseEndpoint.properties.hostName}'
output streamcastUrl string = 'https://${streamcastEndpoint.properties.hostName}'

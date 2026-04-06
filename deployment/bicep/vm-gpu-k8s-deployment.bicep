@description('GUID used for uniqueness')
var uniqueSuffix = uniqueString(resourceGroup().name, resourceGroup().location)

@description('Availability zones for the VMSS')
param zones array = [
  '1'
  '2'
  '3'
]

/*
@description('List of accepted GPU VM sizes')
var acceptedGpuVmSizes = [
  'Standard_ND40rs_v2' // V100
  'Standard_ND96asr_v4' // A100
  'Standard_ND96asr_A100_v4'
  'Standard_NC96ads_A100_v4'
  'Standard_ND96amsr_A100_v4'
  'Standard_ND96isr_H100_v5'
]
*/
// az vm list-skus --all --output table --size H100
@description('GPU VMSS configurations')
var vmssGPUConfigs = [
  // H200
  {
    name: 'h200isr'
    vmSize: 'Standard_ND96isr_H200_v5'
    region: 'westus3'
    zones: zones
    instanceCount: 0 // works
  }
  // H100
  {
    name: 'h100ads'
    vmSize: 'Standard_NC80adis_H100_v5' // 2 x H100 GPUs
    region: 'westus3' // works
    zones: zones
    instanceCount: 1
  }
  {
    name: 'h100isr'
    vmSize: 'Standard_ND96isr_H100_v5' // 8 x H100 GPUs
    region: 'eastus2' // not working
    zones: ['1']
    instanceCount: 1
  }
  /*
  {
    name: 'h100isr'
    vmSize: 'Standard_ND96isr_H100_v5' // 8 x H100 GPUs
    region: 'centraluseuap' // not working
    zones: ['1', '2']
    instanceCount: 1
  }
  {
    name: 'h100isrf'
    // Standard_ND96isf_H100_v5
    // Standard_ND96isr_H100_v5
    vmSize: 'Standard_ND96isrf_H100_v5' // 8 x H100 GPUs
    //region: 'swedencentral' // not working
    region: 'eastus2' // not working
    zones: zones
    instanceCount: 1
  }
  {
    name: 'h100isf'
    vmSize: 'Standard_ND96isf_H100_v5' // 8 x H100 GPUs
    region: 'swedencentral' // not working
    zones: zones
    instanceCount: 0
  }
  {
    name: 'h100isflex'
    vmSize: 'Standard_ND96is_flex_H100_v5' // 8 x H100 GPUs
    region: 'swedencentral' // not working
    zones: zones
    instanceCount: 1
  }
  */
  // A100
  {
    name: 'a100ams'
    vmSize: 'Standard_ND96ams_A100_v4'
    region: 'eastus2'
    zones: zones
    instanceCount: 0 // works
  }
  {
    name: 'a100ads'
    vmSize: 'Standard_NC96ads_A100_v4'
    region: 'westus3'
    zones: zones
    instanceCount: 0
  }
  // V100
  {
    name: 'v100s'
    vmSize: 'Standard_NC24s_v3' // 4 x V100 16 GB
    region: 'southcentralus'
    zones: zones
    instanceCount: 0
  }
  {
    name: 'v100rs'
    vmSize: 'Standard_ND40rs_v2' // 8 x V100 32 GB
    region: 'southcentralus'
    zones: zones
    instanceCount: 0 // works
  }
  // P100
  /*
  {
    name: 'p100s' // Retired VM SKU
    vmSize: 'Standard_NC24rs_v2'
    region: 'westus3'
    zones: zones
    instanceCount: 0 // not available
  }
  */
]

// Azure Container Registry info
@description('Azure Container Registry name')
param acrName string = 'abcd' // TODO fill
@description('Azure Container Registry resource group')
param acrResourceGroup string = 'xyz' // TODO fill

// Hardcoding the GPU VMSS regions for now
@description('List of regions where GPU VMSS are deployed')
var vmssGPURegionIndex = [
  'westus3'
  'swedencentral'
  'eastus2'
  'southcentralus'
  'centraluseuap'
]
// Hardcoding the peering for now (N x (N-1))
@description('List of VNet peering between the GPU VMSS regions')
var vnetPeerings = [
  // westus3
  {
    src: 0
    dst: 1
  }
  {
    src: 0
    dst: 2
  }
  {
    src: 0
    dst: 3
  }
  {
    src: 0
    dst: 4
  }
  // swedencentral
  {
    src: 1
    dst: 0
  }
  {
    src: 1
    dst: 2
  }
  {
    src: 1
    dst: 3
  }
  {
    src: 1
    dst: 4
  }
  // eastus2
  {
    src: 2
    dst: 0
  }
  {
    src: 2
    dst: 1
  }
  {
    src: 2
    dst: 3
  }
  {
    src: 2
    dst: 4
  }
  // southcentralus
  {
    src: 3
    dst: 0
  }
  {
    src: 3
    dst: 1
  }
  {
    src: 3
    dst: 2
  }
  {
    src: 3
    dst: 4
  }
  // centraluseuap
  {
    src: 4
    dst: 0
  }
  {
    src: 4
    dst: 1
  }
  {
    src: 4
    dst: 2
  }
  {
    src: 4
    dst: 3
  }
]


@description('Control plane VMSS VM SKU')
param skuControl string = 'Standard_D16s_v5'

@description('OS publisher for the VMSS')
param ospublisher string = 'Canonical'
@description('OS offer for the VMSS')
param osoffer string = '0001-com-ubuntu-server-focal'
@description('OS SKU for the VMSS')
param ossku string = '20_04-lts-gen2'
@description('OS version for the VMSS')
param osversion string = 'latest'

@description('OS disk size in GB for the VMSS')
@maxValue(1024)
param osDiskSizeInGB int = 256

@description('Data disk size in GB for the VMSS, most VM types already have data drives')
@maxValue(1024)
param dataDiskSizeInGB int = 0 // 256

@description('Admin username for the VMSS')
param adminUsername string = 'azureuser'
@secure()
param adminPassword string = ''
@description('SSH public keys for the VMSS')
param publicKeys array = ['ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDcna8EJSmWpRYVislNdI4uFrx13LVmTbBhbVopkwmTQHRoWGrcH11Ga9+aUko14MmrNEQnagNUVJiyfkXh302VhEiGSaLmPNn+kxQWfQcPhR4TZqaQ2uJw8kZhNcHZxFR5Ylbk0xXX16Qhqm0/mmu9w/OoJjkTgRLGZqbL38vDrbEJ8yUExts0vHLCzVsfgLCkcRQNkOnrIy6pkrBHj2+MTUoWYVKunxxfQJTiGkONe8LIsQtp5hxXgqKbmE17Jb1x37g0GtCJ4j6U8Mo/Su20CIQde77egXt91PedCvf5UEeNH2itkeJ9FP8hk7pYBatvRp1+dZDignPALfIZz8K5']


resource nsg 'Microsoft.Network/networkSecurityGroups@2023-11-01' = {
  name: 'nsg'
  location: resourceGroup().location
  properties: {
    securityRules: [
      {
        name: 'Allow-Incoming-TLS'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '443'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: '*'
          access: 'Allow'
          direction: 'Inbound'
          priority: 100
        }
      }
      {
        name: 'AllowSshRdpOutBound'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          sourceAddressPrefix: '*'
          destinationPortRanges: [
            '22'
            '3389'
          ]
          destinationAddressPrefix: 'VirtualNetwork'
          access: 'Allow'
          priority: 100
          direction: 'Outbound'
        }
      }
      {
        name: 'AllowAzureCloudCommunicationOutBound'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          sourceAddressPrefix: '*'
          destinationPortRange: '443'
          destinationAddressPrefix: 'AzureCloud'
          access: 'Allow'
          priority: 110
          direction: 'Outbound'
        }
      }
      {
        name: 'AllowGetSessionInformationOutBound'
        properties: {
          protocol: '*'
          sourcePortRange: '*'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: 'Internet'
          destinationPortRanges: [
            '80'
            '443'
          ]
          access: 'Allow'
          priority: 130
          direction: 'Outbound'
        }
      }
      {
        name: 'AllowHostPort8181'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          sourceAddressPrefix: '*'
          destinationPortRange: '8181'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 140
          direction: 'Inbound'
        }
      }
      {
        name: 'AllowAzureLoadBalancerPort18181'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          sourceAddressPrefix: 'AzureLoadBalancer'
          destinationPortRange: '18181'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 150
          direction: 'Inbound'
        }
      }
    ]
  }
}

// Create an NSG for each GPU region
resource nsgs 'Microsoft.Network/networkSecurityGroups@2023-11-01' = [
  for (gpuRegion, index) in vmssGPURegionIndex: {
    name: 'nsg-${gpuRegion}'
    location: gpuRegion
    properties: {
      securityRules: [
        {
          name: 'Allow-Incoming-TLS'
          properties: {
            protocol: 'Tcp'
            sourcePortRange: '*'
            destinationPortRange: '443'
            sourceAddressPrefix: '*'
            destinationAddressPrefix: '*'
            access: 'Allow'
            direction: 'Inbound'
            priority: 100
          }
        }
        {
          name: 'AllowSshRdpOutBound'
          properties: {
            protocol: 'Tcp'
            sourcePortRange: '*'
            sourceAddressPrefix: '*'
            destinationPortRanges: [
              '22'
              '3389'
            ]
            destinationAddressPrefix: 'VirtualNetwork'
            access: 'Allow'
            priority: 100
            direction: 'Outbound'
          }
        }
        {
          name: 'AllowAzureCloudCommunicationOutBound'
          properties: {
            protocol: 'Tcp'
            sourcePortRange: '*'
            sourceAddressPrefix: '*'
            destinationPortRange: '443'
            destinationAddressPrefix: 'AzureCloud'
            access: 'Allow'
            priority: 110
            direction: 'Outbound'
          }
        }
        {
          name: 'AllowGetSessionInformationOutBound'
          properties: {
            protocol: '*'
            sourcePortRange: '*'
            sourceAddressPrefix: '*'
            destinationAddressPrefix: 'Internet'
            destinationPortRanges: [
              '80'
              '443'
            ]
            access: 'Allow'
            priority: 130
            direction: 'Outbound'
          }
        }
        {
          name: 'AllowHostPort8181'
          properties: {
            protocol: 'Tcp'
            sourcePortRange: '*'
            sourceAddressPrefix: '*'
            destinationPortRange: '8181'
            destinationAddressPrefix: '*'
            access: 'Allow'
            priority: 140
            direction: 'Inbound'
          }
        }
        {
          name: 'AllowAzureLoadBalancerPort18181'
          properties: {
            protocol: 'Tcp'
            sourcePortRange: '*'
            sourceAddressPrefix: 'AzureLoadBalancer'
            destinationPortRange: '18181'
            destinationAddressPrefix: '*'
            access: 'Allow'
            priority: 150
            direction: 'Inbound'
          }
        }
      ]
    }
  }
]


resource publicIpInternet 'Microsoft.Network/publicIPAddresses@2023-11-01' = {
  name: 'public-ip-internet'
  location: resourceGroup().location
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

resource publicIp 'Microsoft.Network/publicIPAddresses@2023-11-01' = {
  name: 'public-ip-nat'
  location: resourceGroup().location
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

resource publicIps 'Microsoft.Network/publicIPAddresses@2023-11-01' = [
  for (gpuRegion, index) in vmssGPURegionIndex: {
    name: 'public-ip-nat-${gpuRegion}'
    location: gpuRegion
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
]

resource natGateway 'Microsoft.Network/natGateways@2023-11-01' = {
  name: 'nat-gateway'
  location: resourceGroup().location
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIpAddresses: [
      {
        id: publicIp.id
      }
    ]
    idleTimeoutInMinutes: 4
  }
}


resource natGateways 'Microsoft.Network/natGateways@2023-11-01' = [
  for (gpuRegion, index) in vmssGPURegionIndex: {
    name: 'nat-gateway-${gpuRegion}'
    location: gpuRegion
    sku: {
      name: 'Standard'
    }
    properties: {
      publicIpAddresses: [
        {
          id: publicIps[index].id
        }
      ]
      idleTimeoutInMinutes: 4
    }
  }
]

resource vnet 'Microsoft.Network/virtualNetworks@2024-05-01' = {
  name: 'vnet'
  location: resourceGroup().location
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.0.0.0/16'
      ]
    }
    subnets: [
      {
        name: 'LAN'
        properties: {
          addressPrefix: '10.0.0.0/24'
          networkSecurityGroup: {
            id: nsg.id
          }
          defaultOutboundAccess: false
          natGateway: {
            id: natGateway.id
          }
          serviceEndpoints: [
            {
              service: 'Microsoft.KeyVault'
            }
          ]
        }
      }
      {
        name: 'Workers${resourceGroup().location}'
        properties: {
          addressPrefix: '10.0.1.0/24'
          networkSecurityGroup: {
            id: nsg.id
          }
          defaultOutboundAccess: false
          natGateway: {
            id: natGateway.id
          }
          serviceEndpoints: [
            {
              service: 'Microsoft.KeyVault'
            }
          ]
        }
      }
    ]
  }
}

// Create a vnet for the other regions
resource vnets 'Microsoft.Network/virtualNetworks@2024-05-01' = [
  for (gpuRegion, index) in vmssGPURegionIndex: {
    name: 'vnet-${gpuRegion}'
    location: gpuRegion
    properties: {
      addressSpace: {
        addressPrefixes: [
          '10.${index+1}.0.0/16'
        ]
      }
      subnets: [
        {
          name: 'Workers${gpuRegion}'
          properties: {
            addressPrefix: '10.${index+1}.1.0/24'
            networkSecurityGroup: {
              id: nsgs[index].id
            }
            defaultOutboundAccess: false
            natGateway: {
              id: natGateways[index].id
            }
            serviceEndpoints: [
              {
                service: 'Microsoft.KeyVault'
              }
            ]
          }
        }
      ]
    }
  }
]

// Peer the vnets in the other regions with the main vnet
resource vnetPeering1 'Microsoft.Network/virtualNetworks/virtualNetworkPeerings@2024-05-01' = [
  for (gpuRegion, index) in vmssGPURegionIndex: {
    name: 'peer1-${gpuRegion}'
    parent: vnet
    properties: {
      allowVirtualNetworkAccess: true
      allowForwardedTraffic: false // true
      allowGatewayTransit: false
      useRemoteGateways: false
      doNotVerifyRemoteGateways: false
      peerCompleteVnets: true
      remoteVirtualNetwork: {
        id: vnets[index].id
      }
    }
    dependsOn: [
      vnets[index]
    ]
  }
]

resource vnetPeering2 'Microsoft.Network/virtualNetworks/virtualNetworkPeerings@2024-05-01' = [
  for (gpuRegion, index) in vmssGPURegionIndex: {
    name: 'peer2-${gpuRegion}'
    parent:  vnets[index]
    properties: {
      allowVirtualNetworkAccess: true
      allowForwardedTraffic: false // true
      allowGatewayTransit: false
      useRemoteGateways: false
      doNotVerifyRemoteGateways: false
      peerCompleteVnets: true
      remoteVirtualNetwork: {
        id: vnet.id
      }
    }
    dependsOn: [
      vnets[index]
    ]
  }
]

resource vnetPeeringsAll 'Microsoft.Network/virtualNetworks/virtualNetworkPeerings@2024-05-01' = [
  for (peering, index) in vnetPeerings: {
    name: 'peer${peering.src}to${peering.dst}'
    parent: vnets[peering.src]
    properties: {
      allowVirtualNetworkAccess: true
      allowForwardedTraffic: true
      allowGatewayTransit: false
      useRemoteGateways: false
      doNotVerifyRemoteGateways: false
      peerCompleteVnets: true
      remoteVirtualNetwork: {
        id: vnets[peering.dst].id
      }
    }
    dependsOn: [
      vnets[peering.src]
      vnets[peering.dst]
    ]
  }
]


module vmssControlModule 'vmss-linux.bicep' = {
  name: 'vmss-control'
  params: {
    publicKeys: publicKeys 
    subnetId: vnet.properties.subnets[0].id
    zones: zones
    sku: skuControl
    priority: 'Regular' // Control plane should not be spot
    adminUsername: adminUsername
    adminPassword: adminPassword
    ospublisher: ospublisher
    osoffer: osoffer
    ossku: ossku
    osversion: osversion
    osDiskSizeInGB: osDiskSizeInGB
    dataDiskSizeInGB: dataDiskSizeInGB
    cloudinitEncodedString: loadFileAsBase64('cloud-init-control.yml')
    vmssTags: {}
    instanceCount: 1 // Control plane should have only one instance
    resourceNamePrefix: 'control'
    computeNamePrefix: 'control'
    secureBootEnabled: true
    lbBackendPoolIds: [
      {
        id: controlLb.properties.backendAddressPools[0].id
      }
    ]
  }
}

module vmssGPUModules 'vmss-linux.bicep' = [
  for config in vmssGPUConfigs: {
    name: 'vmss-${config.name}'
    params: {
      location: config.region
      publicKeys: publicKeys
      subnetId: vnets[indexOf(vmssGPURegionIndex, config.region)].properties.subnets[0].id
      zones: config.zones
      sku: config.vmSize
      priority: 'Spot'
      adminUsername: adminUsername
      adminPassword: adminPassword
      ospublisher: ospublisher
      osoffer: osoffer
      ossku: ossku
      osversion: osversion
      osDiskSizeInGB: osDiskSizeInGB
      dataDiskSizeInGB: 0 // dataDiskSizeInGB
      cloudinitEncodedString: loadFileAsBase64('cloud-init-gpu.yml')
      vmssTags: {}
      instanceCount: config.instanceCount
      resourceNamePrefix: 'gpu-${config.vmSize}'
      computeNamePrefix: config.name
      secureBootEnabled: false // NVIDIA driver does not support secure boot
      singlePlacementGroup: length(zones) > 1 ? false : true // RDMA requires a single placement group but that cannot happen with multiple zones
    }
  }
]

// Create key vault to store the K8s secrets
resource keyVault 'Microsoft.KeyVault/vaults@2024-12-01-preview' = {
  name: 'vault${uniqueSuffix}'
  location: resourceGroup().location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
      ipRules: [
        {
          value: publicIp.properties.ipAddress
        }
        /*
        for i in range(0, length(vmssGPURegionIndex)): {
          value: reference(publicIps[i].id).ipAddress
        }
        */
        // TODO need to add the public IPs without hardcoding
        {
          value: publicIps[0].properties.ipAddress
        }
        {
          value: publicIps[1].properties.ipAddress
        }
        {
          value: publicIps[2].properties.ipAddress
        }
        {
          value: publicIps[3].properties.ipAddress
        }
      ]
      virtualNetworkRules: [
        {
          id: '${vnet.id}/subnets/LAN'
        }
        {
          id: '${vnet.id}/subnets/Workers${resourceGroup().location}'
        }
        // TODO need to add the public IPs without hardcoding
        {
          id: '${vnets[0].id}/subnets/Workers${vmssGPURegionIndex[0]}'
        }
        {
          id: '${vnets[1].id}/subnets/Workers${vmssGPURegionIndex[1]}'
        }
        {
          id: '${vnets[2].id}/subnets/Workers${vmssGPURegionIndex[2]}'
        }
        {
          id: '${vnets[3].id}/subnets/Workers${vmssGPURegionIndex[3]}'
        }
      ]
    }
    accessPolicies: []
    enabledForDeployment: false
    enabledForTemplateDeployment: true
    enableSoftDelete: false
    enableRbacAuthorization: true
    publicNetworkAccess: 'Enabled'
  }
}

resource roleKVAssignmentControl 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, 'control', 'control_secrets_writer')
  scope: keyVault
  properties: {
    principalId: vmssControlModule.outputs.principalId
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      'b86a8fe4-44ce-4948-aee5-eccb2c155cd7') // Key Vault Secrets Officer
  }
}

resource roleKVAssignmentGPU 'Microsoft.Authorization/roleAssignments@2022-04-01' = [
  for (config, i) in vmssGPUConfigs: {
    name: guid(keyVault.id, config.name, config.region, 'gpu_secrets_writer')
    scope: keyVault
    properties: {
      principalId: vmssGPUModules[i].outputs.principalId
      roleDefinitionId: subscriptionResourceId(
        'Microsoft.Authorization/roleDefinitions',
        'b86a8fe4-44ce-4948-aee5-eccb2c155cd7' // Key Vault Secrets Officer
      )
    }
    dependsOn: [
      vmssGPUModules[i]
    ]
  }
]

// Let the VMSS pull from the ACR
module acrRoleAssignment 'roleACRAssignment.bicep' = {
  name: 'acr-role-control'
  scope: resourceGroup(acrResourceGroup)
  params: {
    principalId: vmssControlModule.outputs.principalId
    acrName: acrName
  }
}

module acrRoleAssignments 'roleACRAssignment.bicep' = [
  for (config, i) in vmssGPUConfigs: {
    name: 'acr-role-${config.name}'
    scope: resourceGroup(acrResourceGroup)
    params: {
      principalId: vmssGPUModules[i].outputs.principalId
      acrName: acrName
    }
  }
]


// TODO add storage and containers
// https://dev.azure.com/azsr/AzureDeploy/_git/AzTemplates?path=/bicep/resources/storage-account.bicep


// Load balancer
resource controlLb 'Microsoft.Network/loadBalancers@2023-11-01' = {
  name: 'control-lb'
  location: resourceGroup().location
  sku: {
    name: 'Standard'
  }
  properties: {
    frontendIPConfigurations: [
      {
        name: 'LoadBalancerFrontEnd'
        properties: {
          publicIPAddress: {
            id: publicIpInternet.id
          }
        }
      }
    ]
    backendAddressPools: [
      {
        name: 'controlBackendPool'
      }
    ]
    loadBalancingRules: [
      {
        name: 'LBRule-18181'
        properties: {
          frontendIPConfiguration: {
            id: resourceId('Microsoft.Network/loadBalancers/frontendIPConfigurations', 'control-lb', 'LoadBalancerFrontEnd')
          }
          backendAddressPool: {
            id: resourceId('Microsoft.Network/loadBalancers/backendAddressPools', 'control-lb', 'controlBackendPool')
          }
          protocol: 'Tcp'
          frontendPort: 18181
          backendPort: 18181
          enableFloatingIP: false
          idleTimeoutInMinutes: 4
          loadDistribution: 'Default'
          probe: {
            id: resourceId('Microsoft.Network/loadBalancers/probes', 'control-lb', 'controlHealthProbe')
          }
        }
      }
    ]
    probes: [
      {
        name: 'controlHealthProbe'
        properties: {
          protocol: 'Tcp'
          port: 18181
          intervalInSeconds: 5
          numberOfProbes: 2
        }
      }
    ]
  }
}

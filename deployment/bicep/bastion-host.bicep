param subnetid string

param resourceNamePrefix string = ''
var location = resourceGroup().location

var namePrefix = empty(resourceNamePrefix) ? resourceGroup().name : resourceNamePrefix

// Create a public IP address for the bastion host.
resource publicIp 'Microsoft.Network/publicIPAddresses@2023-11-01' = {
  name: 'public-ip-bastion'
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


// Create bastion host.
resource bastionHost 'Microsoft.Network/bastionHosts@2023-11-01' = {
  name: '${namePrefix}-bastion-host'
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    enableTunneling:true
    enableIpConnect:true
    scaleUnits: 2
    ipConfigurations: [
      {
        name: 'IpConf'
        properties: {
          subnet: {
            id: subnetid
          }
          publicIPAddress: {
            id: publicIp.id
          }
        }
      }
    ]
  }
}

output bastionPublicIpAddress string = publicIp.properties.ipAddress

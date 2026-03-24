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
      }
    ]
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
// Outputs
// ---------------------------------------------------------------------------
output clusterName string = aksCluster.name
output publicIpAddress string = publicIp.properties.ipAddress
output publicIpName string = publicIp.name
output gpuNodePoolName string = gpuNodePool.name
output gpuMigNodePoolName string = gpuMigNodePool.name

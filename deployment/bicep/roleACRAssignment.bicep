@description('Name of the ACR')
param acrName string

@description('Principal ID to assign AcrPull role')
param principalId string

resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' existing = {
  name: acrName
}

resource acrPullRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, 'acdd72a7-3385-48ef-bd42-f606fba81ae7') // AcrPull roleDefinitionId
  scope: acr
  properties: {
    principalId: principalId
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      'acdd72a7-3385-48ef-bd42-f606fba81ae7' // AcrPull
    )
  }
}

param name string
param location string = resourceGroup().location
param tags object = {}

param identityName string
param containerAppsEnvironmentName string
param containerRegistryName string
param exists bool
@secure()
param appDefinition object = {}

param applicationInsightsName string = ''
param keyVaultName string = ''

resource userIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
}

resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' existing = {
  name: containerAppsEnvironmentName
}

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: containerRegistryName
}

resource applicationInsights 'Microsoft.Insights/components@2020-02-02' existing = if (!empty(applicationInsightsName)) {
  name: applicationInsightsName
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' existing = if (!empty(keyVaultName)) {
  name: keyVaultName
}

resource app 'Microsoft.App/containerApps@2023-05-01' = {
  name: name
  location: location
  tags: union(tags, {'azd-service-name':  'app'})
  dependsOn: [
    userIdentity
  ]
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${userIdentity.id}': {} }
  }
  properties: {
    managedEnvironmentId: containerAppsEnvironment.id
    configuration: {
      activeRevisionsMode: 'single'
      ingress:  {
        external: true
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
      }
      registries: [
        {
          server: '${containerRegistry.name}.azurecr.io'
          identity: userIdentity.id
        }
      ]
      secrets: [
        {
          name: 'registry-password'
          value: containerRegistry.listCredentials().passwords[0].value
        }
      ]
    }
    template: {
      containers: [
        {
          image: !exists ? 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest' : 'nginx:latest'
          name: 'main'
          env: [
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              value: !empty(applicationInsightsName) ? applicationInsights.properties.ConnectionString : ''
            }
            {
              name: 'PYTHONUNBUFFERED'
              value: '1'
            }
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: 'https://jb-demo-foundry-project-resource.openai.azure.com/'
            }
            {
              name: 'AZURE_OPENAI_MODEL_DEPLOYMENT'
              value: 'gpt-4.1'
            }
            {
              name: 'AZURE_OPENAI_API_VERSION'
              value: '2025-01-01-preview'
            }
            {
              name: 'AZURE_SEARCH_SERVICE_ENDPOINT'
              value: 'https://aisearch-basic-eastus.search.windows.net'
            }
            {
              name: 'AZURE_SEARCH_INDEX_NAME'
              value: 'farmandfleetdemo-social'
            }
            {
              name: 'AZURE_SEARCH_SURVEY_INDEX_NAME'
              value: 'farmandfleetdemo-surveys'
            }
            {
              name: 'SQL_SERVER'
              value: 'amaharajcloudstrat.database.windows.net'
            }
            {
              name: 'SQL_PORT'
              value: '1433'
            }
            {
              name: 'SQL_DATABASE'
              value: 'sql-amaharaj-cloudstrat'
            }
            {
              name: 'SQL_SCHEMA'
              value: 'dbo'
            }
            {
              name: 'SQL_USERNAME'
              value: 'bsff'
            }
            {
              name: 'SQL_TABLES'
              value: 'MEDALLIA_FEEDBACK2'
            }
            {
              name: 'SQL_DATA_DICTIONARY_PATH'
              value: 'src/plugins/sqldb/data_dictionaries/'
            }
          ]
          resources: {
            cpu: json('1.0')
            memory: '2.0Gi'
          }
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 10
      }
    }
  }
}

// Grant the identity with Contributor role permissions over the resource group so it can run AZD
resource identityRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(resourceGroup().id, userIdentity.id, subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'b24988ac-6180-42a0-ab88-20f7382dd24c'))
  scope: resourceGroup()
  properties: {
    principalId: userIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId:  subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'b24988ac-6180-42a0-ab88-20f7382dd24c')
  }
}

// Grant the identity with AcrPull role permissions over the registry so it can pull images
resource registryRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(containerRegistry.id, userIdentity.id, subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d'))
  scope: containerRegistry
  properties: {
    principalId: userIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId:  subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
  }
}

// Grant the identity with Key Vault Secrets User role permissions so it can read secrets
resource keyVaultRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(keyVaultName)) {
  name: guid(keyVault.id, userIdentity.id, subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6'))
  scope: keyVault
  properties: {
    principalId: userIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId:  subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
  }
}

output defaultDomain string = containerAppsEnvironment.properties.defaultDomain
output name string = app.name
output uri string = 'https://${app.properties.configuration.ingress.fqdn}'
output id string = app.id
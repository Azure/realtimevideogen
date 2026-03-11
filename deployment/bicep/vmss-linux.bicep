param location string = resourceGroup().location

param resourceNamePrefix string = ''

var namePrefix = empty(resourceNamePrefix) ? resourceGroup().name : resourceNamePrefix

param computeNamePrefix string = 'vm'

param zones array = ['1']

param sku string = 'Standard_DS2_v2'

param ospublisher string = 'MicrosoftCblMariner'
param osoffer string = 'Cbl-Mariner'
param ossku string = 'cbl-mariner-2-Gen2'
param osversion string = 'latest'

@maxValue(1024)
param osDiskSizeInGB int = 256

// max 1T, i think.
@maxValue(1024)
param dataDiskSizeInGB int = 256

param subnetId string

// Regular, Spot, Low
@allowed([
  'Regular'
  'Spot'
  'Low'
])
param priority string = 'Regular'

param adminUsername string = 'azrsadmin'
@secure()
param adminPassword string = ''
param publicKeys array = []

var publicKeysObject = [for publicKey in publicKeys: {
  keyData: publicKey
  path: '/home/${adminUsername}/.ssh/authorized_keys'
}]
              
// Custom script to run after VM deployment of OS Upgrade.
// This is an base64 encoded string of a shell script.
param bashScriptEncodedString string = ''

// Cloud-init script to run after VM deployment.
// This is an base64 encoded string of a cloud init script. 
param cloudinitEncodedString string = ''

param instanceCount int = 1

param vmssTags object = {}

param accelNetEnabled bool = true
param lbBackendPoolIds array = []

param secureBootEnabled bool = true
param vTpmEnabled bool = true

param singlePlacementGroup bool = false

var defaultTags = {
  'platformsettings.host_environment.service.platform_optedin_for_rootcerts': 'true'
  azsecpack: 'nonprod'
}

var completeVmssTags = union(defaultTags, vmssTags)



resource vmss 'Microsoft.Compute/virtualMachineScaleSets@2024-03-01' = {
  name: '${namePrefix}-vmss'
  location: location
  zones: zones
  tags: completeVmssTags 
  sku: {
    name: sku
    capacity: instanceCount 
    tier: 'Standard'
  }
  // Managed Identity of the scale set to access other resources.
  // System identity is needed for AADSSHLogin extension.
  identity: {
    type: 'SystemAssigned' // 'None' | 'SystemAssigned' | 'UserAssigned' | 'SystemAssigned, UserAssigned'
  }

  properties: {
    // Automatic Repair Policy: enabled, 30 min grace period, using Restart action. 
    automaticRepairsPolicy: {
      enabled: true 
      gracePeriod: 'PT30M'
      repairAction: 'Restart'
    }
    orchestrationMode	: 'Uniform'
    overprovision: false
    platformFaultDomainCount: 1
    singlePlacementGroup: singlePlacementGroup // max scale 100 when true
    upgradePolicy:{
      mode:'Automatic'
      automaticOSUpgradePolicy:{
        enableAutomaticOSUpgrade:true
      }
    }
    virtualMachineProfile: {
      //boot diagnostics
      diagnosticsProfile: {
        bootDiagnostics: {
          enabled: true
        }
      }
      extensionProfile: {
        extensions: [
          {
            // This is the health extension for the VMSS.
            // This is required for Auto OS image Upgrade to work.
            // Here, we probe the port 22 to check health status.
            // If a handshake is successful, the VM is considered healthy.
            // And OS upgrade will be triggered only if VM is healthy.
            name: '${namePrefix}-vmss-health-extension'
            properties: {
              publisher: 'Microsoft.ManagedServices'
              type: 'ApplicationHealthLinux'
              typeHandlerVersion: '1.0'
              autoUpgradeMinorVersion: true
              enableAutomaticUpgrade: true
              settings: {
                protocol: 'tcp'
                port: 22
              }
            }
          }
          {
            // This is the Azure Monitor Linux Extension
            // This is required for AzSecPack to work.
            name: '${namePrefix}-vmss-azure-monitor'
            properties: {
              publisher: 'Microsoft.Azure.Monitor'
              type: 'AzureMonitorLinuxAgent'
              typeHandlerVersion: '1.0'
              autoUpgradeMinorVersion: true
              enableAutomaticUpgrade: true
              settings: {
                GCS_AUTO_CONFIG: true
              }
            }
          }
          {
            // This is the Azure Monitor Linux Extension
            // This is required for AzSecPack to work.
            name: '${namePrefix}-vmss-azure-security-monitoring'
            properties: {
              publisher: 'Microsoft.Azure.Security.Monitoring'
              type: 'AzureSecurityLinuxAgent'
              typeHandlerVersion: '2.0'
              autoUpgradeMinorVersion: true
              enableAutomaticUpgrade: true
              settings: {
                enableGenevaUpload:true
                enableAutoConfig:true
              }
            }
          }
          {
            // This is the Azure AD SSH Login Extension
            // Used for SSH login to the VMSS using Azure AD credentials.
            // Without the need for sharing password or keys.
            name: '${namePrefix}-vmss-aadsshlogin'
            properties: {
              publisher: 'Microsoft.Azure.ActiveDirectory'
              type: 'AADSSHLoginForLinux'
              typeHandlerVersion: '1.0'
            }
          }
        ]
      }
      networkProfile: {
        networkInterfaceConfigurations: [
          {
            name: 'nicconfig'
            properties: {
              enableAcceleratedNetworking: accelNetEnabled
              primary: true
              ipConfigurations: [
                {
                  name: 'ipconfig'
                  properties: {
                    subnet: {
                      id: subnetId
                    }
                    loadBalancerBackendAddressPools: lbBackendPoolIds
                  }
                }
              ]
            }
          }
        ]
      }
      osProfile: {
        computerNamePrefix: computeNamePrefix
        adminUsername: adminUsername
        adminPassword: adminPassword
        customData: cloudinitEncodedString
        linuxConfiguration: {
          disablePasswordAuthentication: false
          // This is required for Extensions to work.
          provisionVMAgent: true
          ssh: {    
            publicKeys: publicKeysObject 
          }
        }
      }
      priority: priority
      storageProfile: {
        imageReference: {
          publisher: ospublisher
          offer: osoffer
          sku: ossku
          version: osversion
        }
        osDisk: {
          createOption: 'FromImage'
          diskSizeGB: osDiskSizeInGB
          managedDisk: {
            storageAccountType: 'Premium_LRS'
          }
        }
        dataDisks: (dataDiskSizeInGB > 0) ? [
          {
            createOption: 'Empty'
            diskSizeGB: dataDiskSizeInGB
            lun: 0
            managedDisk: {
              storageAccountType: 'Premium_LRS'
            }
          }
        ] : []
      }
      securityProfile: {
        securityType: 'TrustedLaunch'
        uefiSettings: {
          secureBootEnabled: secureBootEnabled
          vTpmEnabled: vTpmEnabled
        }
      }
    }
  }
}

// Custom Script Extension
// This is used to run a custom script after the VM deployment.
// This is useful for running post-deployment tasks like installing software, etc.
resource customScriptExtension 'Microsoft.Compute/virtualMachineScaleSets/extensions@2024-03-01' = if (bashScriptEncodedString != '') {
  name: '${namePrefix}-vmss-custom-script'
  parent: vmss
  properties: {
    publisher: 'Microsoft.Azure.Extensions' 
    type: 'CustomScript'
    typeHandlerVersion: '2.1'
    autoUpgradeMinorVersion: true
    settings: {

    }
    protectedSettings: {
      script: bashScriptEncodedString 
    }
  }
}

output principalId string = vmss.identity.principalId

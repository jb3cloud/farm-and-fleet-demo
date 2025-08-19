#!/usr/bin/env pwsh

# Azure Developer CLI Pre-deployment Hook
# This script prompts for secrets and stores them in Azure Key Vault
# before deploying the application to ensure all secrets are available

param(
    [Parameter(Mandatory = $true)]
    [string]$KeyVaultName,
    
    [Parameter(Mandatory = $false)]
    [string]$EnvironmentName = "farmandfleetdemo"
)

Write-Host "🔐 Setting up secrets in Azure Key Vault..." -ForegroundColor Yellow

# Define the secrets we need to collect
$secrets = @{
    "AZURE-OPENAI-API-KEY" = @{
        "prompt" = "Enter your Azure OpenAI API Key"
        "description" = "API key for Azure OpenAI service"
        "required" = $true
    }
    "AZURE-SEARCH-API-KEY" = @{
        "prompt" = "Enter your Azure Search API Key"
        "description" = "API key for Azure Cognitive Search service"
        "required" = $true
    }
    "SQL-PASSWORD" = @{
        "prompt" = "Enter your SQL Database Password"
        "description" = "Password for SQL Server authentication"
        "required" = $true
    }
    "AZURE-LANGUAGE-KEY" = @{
        "prompt" = "Enter your Azure Language Service Key (optional)"
        "description" = "API key for Azure Language Service"
        "required" = $false
    }
    "AZURE-OPENAI-EMBEDDING-DEPLOYMENT" = @{
        "prompt" = "Enter your Azure OpenAI Embedding Deployment Name (optional)"
        "description" = "Name of the text embedding deployment"
        "required" = $false
        "default" = "text-embedding-3-small"
    }
}

# Function to securely prompt for password
function Get-SecureInput {
    param([string]$Prompt, [bool]$IsSecret = $true)
    
    if ($IsSecret) {
        $secureString = Read-Host -Prompt $Prompt -AsSecureString
        $bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureString)
        $value = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
        [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        return $value
    } else {
        return Read-Host -Prompt $Prompt
    }
}

# Check if Azure CLI is logged in
try {
    $account = az account show --query "name" -o tsv 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Please login to Azure CLI first: az login" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Authenticated as: $account" -ForegroundColor Green
} catch {
    Write-Host "❌ Azure CLI not found or not logged in" -ForegroundColor Red
    exit 1
}

Write-Host "`n📋 This script will collect the following secrets:" -ForegroundColor Cyan
foreach ($secretName in $secrets.Keys) {
    $secret = $secrets[$secretName]
    $requiredText = if ($secret.required) { "(Required)" } else { "(Optional)" }
    Write-Host "  • $($secret.description) $requiredText" -ForegroundColor White
}

Write-Host "`n🔍 Checking existing secrets in Key Vault '$KeyVaultName'..." -ForegroundColor Yellow

# Check which secrets already exist
$existingSecrets = @{}
foreach ($secretName in $secrets.Keys) {
    try {
        $existingValue = az keyvault secret show --vault-name $KeyVaultName --name $secretName --query "value" -o tsv 2>$null
        if ($LASTEXITCODE -eq 0 -and ![string]::IsNullOrWhiteSpace($existingValue)) {
            $existingSecrets[$secretName] = $true
            Write-Host "  ✅ $secretName already exists" -ForegroundColor Green
        }
    } catch {
        # Secret doesn't exist, we'll need to set it
    }
}

Write-Host "`n🎯 Collecting missing or updated secrets..." -ForegroundColor Yellow

# Collect secrets from user
$collectedSecrets = @{}
foreach ($secretName in $secrets.Keys) {
    $secret = $secrets[$secretName]
    $hasExisting = $existingSecrets.ContainsKey($secretName)
    
    if ($hasExisting) {
        $response = Read-Host "Secret '$secretName' exists. Update it? [y/N]"
        if ($response.ToLower() -ne 'y') {
            continue
        }
    }
    
    if ($secret.required -or $hasExisting -or (Read-Host "Set optional secret '$secretName'? [y/N]").ToLower() -eq 'y') {
        do {
            Write-Host "`n$($secret.description):" -ForegroundColor Cyan
            $value = Get-SecureInput -Prompt $secret.prompt -IsSecret ($secretName -like "*KEY*" -or $secretName -like "*PASSWORD*")
            
            if ([string]::IsNullOrWhiteSpace($value) -and $secret.ContainsKey("default")) {
                $value = $secret.default
                Write-Host "Using default value: $value" -ForegroundColor Gray
            }
            
            if ([string]::IsNullOrWhiteSpace($value) -and $secret.required) {
                Write-Host "❌ This secret is required. Please provide a value." -ForegroundColor Red
            } else {
                $collectedSecrets[$secretName] = $value
                break
            }
        } while ($secret.required)
    }
}

if ($collectedSecrets.Count -eq 0) {
    Write-Host "`n✅ No secrets to update. Continuing with deployment..." -ForegroundColor Green
    exit 0
}

Write-Host "`n🔐 Storing secrets in Azure Key Vault '$KeyVaultName'..." -ForegroundColor Yellow

# Store secrets in Key Vault
$successCount = 0
$errorCount = 0

foreach ($secretName in $collectedSecrets.Keys) {
    $value = $collectedSecrets[$secretName]
    try {
        Write-Host "  Setting $secretName..." -ForegroundColor White
        az keyvault secret set --vault-name $KeyVaultName --name $secretName --value $value --output none
        if ($LASTEXITCODE -eq 0) {
            $successCount++
            Write-Host "    ✅ Success" -ForegroundColor Green
        } else {
            $errorCount++
            Write-Host "    ❌ Failed" -ForegroundColor Red
        }
    } catch {
        $errorCount++
        Write-Host "    ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`n📊 Results:" -ForegroundColor Cyan
Write-Host "  ✅ Successfully set: $successCount secrets" -ForegroundColor Green
if ($errorCount -gt 0) {
    Write-Host "  ❌ Failed to set: $errorCount secrets" -ForegroundColor Red
    Write-Host "  💡 Check Key Vault permissions and try again" -ForegroundColor Yellow
}

if ($errorCount -eq 0) {
    Write-Host "`n🎉 All secrets configured successfully! Deployment will proceed..." -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n⚠️  Some secrets failed to set. Check the errors above." -ForegroundColor Yellow
    $response = Read-Host "Continue with deployment anyway? [y/N]"
    if ($response.ToLower() -eq 'y') {
        exit 0
    } else {
        Write-Host "❌ Deployment cancelled by user" -ForegroundColor Red
        exit 1
    }
}
#!/bin/bash

# Azure Developer CLI Pre-deployment Hook
# This script prompts for secrets and stores them in Azure Key Vault
# before deploying the application to ensure all secrets are available

set -e  # Exit on any error

KEYVAULT_NAME="$1"
ENVIRONMENT_NAME="${2:-farmandfleetdemo}"

if [ -z "$KEYVAULT_NAME" ]; then
    echo "❌ Key Vault name is required as first parameter"
    exit 1
fi

echo "🔐 Setting up secrets in Azure Key Vault..."

# Define the secrets we need to collect (using arrays instead of associative arrays for compatibility)
SECRET_NAMES=(
    "AZURE-OPENAI-API-KEY"
    "AZURE-SEARCH-API-KEY" 
    "SQL-PASSWORD"
    "AZURE-LANGUAGE-KEY"
    "AZURE-OPENAI-EMBEDDING-DEPLOYMENT"
)

SECRET_PROMPTS=(
    "Enter your Azure OpenAI API Key"
    "Enter your Azure Search API Key"
    "Enter your SQL Database Password"
    "Enter your Azure Language Service Key (optional)"
    "Enter your Azure OpenAI Embedding Deployment Name (optional)"
)

SECRET_DESCRIPTIONS=(
    "API key for Azure OpenAI service"
    "API key for Azure Cognitive Search service"
    "Password for SQL Server authentication"
    "API key for Azure Language Service"
    "Name of the text embedding deployment"
)

SECRET_REQUIRED=(
    "required"
    "required"
    "required"
    "optional"
    "optional"
)

SECRET_DEFAULTS=(
    ""
    ""
    ""
    ""
    "text-embedding-3-small"
)

# Function to securely prompt for input
get_secret_input() {
    local prompt="$1"
    local is_secret="$2"
    
    if [ "$is_secret" = "true" ]; then
        echo -n "$prompt: " >&2
        read -s value
        echo >&2  # New line after hidden input
    else
        echo -n "$prompt: " >&2
        read value
    fi
    echo "$value"
}

# Check if Azure CLI is logged in
echo "🔍 Checking Azure CLI authentication..."
if ! az account show >/dev/null 2>&1; then
    echo "❌ Please login to Azure CLI first: az login"
    exit 1
fi

account_name=$(az account show --query "name" -o tsv)
echo "✅ Authenticated as: $account_name"

echo ""
echo "📋 This script will collect the following secrets:"
for i in "${!SECRET_NAMES[@]}"; do
    secret_name="${SECRET_NAMES[$i]}"
    description="${SECRET_DESCRIPTIONS[$i]}"
    required_status="${SECRET_REQUIRED[$i]}"
    required_text="(Required)"
    if [ "$required_status" = "optional" ]; then
        required_text="(Optional)"
    fi
    echo "  • $description $required_text"
done

echo ""
echo "🔍 Checking existing secrets in Key Vault '$KEYVAULT_NAME'..."

# Check which secrets already exist
existing_secrets=()
for secret_name in "${SECRET_NAMES[@]}"; do
    if az keyvault secret show --vault-name "$KEYVAULT_NAME" --name "$secret_name" >/dev/null 2>&1; then
        existing_secrets+=("$secret_name")
        echo "  ✅ $secret_name already exists"
    fi
done

echo ""
echo "🎯 Collecting missing or updated secrets..."

# Function to check if secret exists
secret_exists() {
    local secret_name="$1"
    for existing in "${existing_secrets[@]}"; do
        if [ "$existing" = "$secret_name" ]; then
            return 0
        fi
    done
    return 1
}

# Collect secrets from user
collected_secrets=()
collected_values=()
for i in "${!SECRET_NAMES[@]}"; do
    secret_name="${SECRET_NAMES[$i]}"
    prompt="${SECRET_PROMPTS[$i]}"
    description="${SECRET_DESCRIPTIONS[$i]}"
    required_status="${SECRET_REQUIRED[$i]}"
    default_value="${SECRET_DEFAULTS[$i]}"
    
    has_existing=false
    if secret_exists "$secret_name"; then
        has_existing=true
    fi
    
    if [ "$has_existing" = "true" ]; then
        echo -n "Secret '$secret_name' exists. Update it? [y/N]: "
        read -r response
        if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
            continue
        fi
    fi
    
    should_collect="false"
    if [ "$required_status" = "required" ] || [ "$has_existing" = "true" ]; then
        should_collect="true"
    else
        echo -n "Set optional secret '$secret_name'? [y/N]: "
        read -r response
        if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
            should_collect="true"
        fi
    fi
    
    if [ "$should_collect" = "true" ]; then
        echo ""
        echo "$description:"
        
        while true; do
            is_secret="false"
            if [[ "$secret_name" == *"KEY"* ]] || [[ "$secret_name" == *"PASSWORD"* ]]; then
                is_secret="true"
            fi
            
            value=$(get_secret_input "$prompt" "$is_secret")
            
            if [ -z "$value" ] && [ -n "$default_value" ]; then
                value="$default_value"
                echo "Using default value: $value"
            fi
            
            if [ -z "$value" ] && [ "$required_status" = "required" ]; then
                echo "❌ This secret is required. Please provide a value."
            else
                collected_secrets+=("$secret_name")
                collected_values+=("$value")
                break
            fi
        done
    fi
done

if [ ${#collected_secrets[@]} -eq 0 ]; then
    echo ""
    echo "✅ No secrets to update. Continuing with deployment..."
    exit 0
fi

echo ""
echo "🔐 Storing secrets in Azure Key Vault '$KEYVAULT_NAME'..."

# Store secrets in Key Vault
success_count=0
error_count=0

for i in "${!collected_secrets[@]}"; do
    secret_name="${collected_secrets[$i]}"
    value="${collected_values[$i]}"
    echo "  Setting $secret_name..."
    
    if az keyvault secret set --vault-name "$KEYVAULT_NAME" --name "$secret_name" --value "$value" --output none 2>/dev/null; then
        ((success_count++))
        echo "    ✅ Success"
    else
        ((error_count++))
        echo "    ❌ Failed"
    fi
done

echo ""
echo "📊 Results:"
echo "  ✅ Successfully set: $success_count secrets"
if [ $error_count -gt 0 ]; then
    echo "  ❌ Failed to set: $error_count secrets"
    echo "  💡 Check Key Vault permissions and try again"
fi

if [ $error_count -eq 0 ]; then
    echo ""
    echo "🎉 All secrets configured successfully! Deployment will proceed..."
    exit 0
else
    echo ""
    echo "⚠️  Some secrets failed to set. Check the errors above."
    echo -n "Continue with deployment anyway? [y/N]: "
    read -r response
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        exit 0
    else
        echo "❌ Deployment cancelled by user"
        exit 1
    fi
fi
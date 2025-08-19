#!/bin/bash

# Simple script to set secrets in Key Vault using the secrets file
set -e

KEYVAULT_NAME="${AZURE_KEY_VAULT_NAME}"
SECRETS_FILE="secrets"

if [ -z "$KEYVAULT_NAME" ]; then
    echo "❌ AZURE_KEY_VAULT_NAME environment variable is required"
    exit 1
fi

if [ ! -f "$SECRETS_FILE" ]; then
    echo "❌ Secrets file not found: $SECRETS_FILE"
    exit 1
fi

echo "🔐 Setting secrets from $SECRETS_FILE into Key Vault '$KEYVAULT_NAME'..."

# Check if Azure CLI is logged in
if ! az account show >/dev/null 2>&1; then
    echo "❌ Please login to Azure CLI first: az login"
    exit 1
fi

success_count=0
error_count=0

# Read secrets file and set each secret
while IFS='=' read -r key value; do
    # Skip empty lines and comments
    [[ -z "$key" || "$key" =~ ^#.*$ ]] && continue
    
    # Convert key to Key Vault format (replace underscores with hyphens)
    kv_key=$(echo "$key" | tr '_' '-')
    
    echo "  Setting $kv_key..."
    if az keyvault secret set --vault-name "$KEYVAULT_NAME" --name "$kv_key" --value "$value" --output none 2>/dev/null; then
        ((success_count++))
        echo "    ✅ Success"
    else
        ((error_count++))
        echo "    ❌ Failed"
    fi
done < "$SECRETS_FILE"

echo ""
echo "📊 Results:"
echo "  ✅ Successfully set: $success_count secrets"
if [ $error_count -gt 0 ]; then
    echo "  ❌ Failed to set: $error_count secrets"
fi

echo ""
echo "🎉 Secrets configuration complete!"
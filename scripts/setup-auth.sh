#!/bin/bash
# Setup script for basic authentication

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NGINX_DIR="$PROJECT_DIR/nginx"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== SoraWM Authentication Setup ===${NC}"

# Check if htpasswd is available
if ! command -v htpasswd &> /dev/null; then
    echo -e "${YELLOW}htpasswd not found. Installing apache2-utils...${NC}"
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y apache2-utils
    elif command -v brew &> /dev/null; then
        brew install httpd
    else
        echo -e "${RED}Please install htpasswd manually${NC}"
        exit 1
    fi
fi

# Create nginx directory if it doesn't exist
mkdir -p "$NGINX_DIR"

# Get username
read -p "Enter username: " USERNAME
if [ -z "$USERNAME" ]; then
    echo -e "${RED}Username cannot be empty${NC}"
    exit 1
fi

# Get password
read -s -p "Enter password: " PASSWORD
echo
if [ -z "$PASSWORD" ]; then
    echo -e "${RED}Password cannot be empty${NC}"
    exit 1
fi

read -s -p "Confirm password: " PASSWORD_CONFIRM
echo
if [ "$PASSWORD" != "$PASSWORD_CONFIRM" ]; then
    echo -e "${RED}Passwords do not match${NC}"
    exit 1
fi

# Create or update .htpasswd file
HTPASSWD_FILE="$NGINX_DIR/.htpasswd"

if [ -f "$HTPASSWD_FILE" ]; then
    # Update existing file
    htpasswd -b "$HTPASSWD_FILE" "$USERNAME" "$PASSWORD"
    echo -e "${GREEN}Updated user '$USERNAME' in $HTPASSWD_FILE${NC}"
else
    # Create new file
    htpasswd -bc "$HTPASSWD_FILE" "$USERNAME" "$PASSWORD"
    echo -e "${GREEN}Created $HTPASSWD_FILE with user '$USERNAME'${NC}"
fi

# Set proper permissions
chmod 644 "$HTPASSWD_FILE"

echo -e "${GREEN}Authentication setup complete!${NC}"
echo -e "${YELLOW}You can add more users by running this script again.${NC}"

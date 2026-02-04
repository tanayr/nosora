#!/bin/bash
# One-click installation script for GPU servers (DigitalOcean, etc.)
# Run this on a fresh Ubuntu 22.04 server with NVIDIA GPU

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== SoraWM GPU Server Installation ===${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}Please run as root (sudo)${NC}"
    exit 1
fi

# Update system
echo -e "${GREEN}Updating system packages...${NC}"
apt-get update && apt-get upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo -e "${GREEN}Installing Docker...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    systemctl enable docker
    systemctl start docker
fi

# Install Docker Compose plugin
if ! docker compose version &> /dev/null; then
    echo -e "${GREEN}Installing Docker Compose...${NC}"
    apt-get install -y docker-compose-plugin
fi

# Install NVIDIA Container Toolkit
if ! command -v nvidia-container-toolkit &> /dev/null; then
    echo -e "${GREEN}Installing NVIDIA Container Toolkit...${NC}"
    
    # Add NVIDIA GPG key and repository
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    apt-get update
    apt-get install -y nvidia-container-toolkit
    
    # Configure Docker to use NVIDIA runtime
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

# Verify GPU is accessible
echo -e "${GREEN}Verifying GPU access...${NC}"
if nvidia-smi &> /dev/null; then
    nvidia-smi
    echo -e "${GREEN}GPU detected successfully!${NC}"
else
    echo -e "${RED}Warning: GPU not detected. Please ensure NVIDIA drivers are installed.${NC}"
fi

# Install htpasswd for authentication
apt-get install -y apache2-utils

# Clone or update repository
INSTALL_DIR="/opt/sorawm"
if [ -d "$INSTALL_DIR" ]; then
    echo -e "${YELLOW}Updating existing installation...${NC}"
    cd "$INSTALL_DIR"
    git pull
else
    echo -e "${GREEN}Cloning repository...${NC}"
    git clone https://github.com/linkedlist771/SoraWatermarkCleaner.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Create necessary directories
mkdir -p nginx/ssl working_dir logs resources/checkpoint

# Setup authentication if not exists
if [ ! -f "nginx/.htpasswd" ]; then
    echo -e "${YELLOW}Setting up authentication...${NC}"
    read -p "Enter admin username: " ADMIN_USER
    htpasswd -c nginx/.htpasswd "$ADMIN_USER"
fi

# Build and start services
echo -e "${GREEN}Building Docker images (this may take a while)...${NC}"
docker compose build

echo -e "${GREEN}Starting services...${NC}"
docker compose up -d

# Get server IP
SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')

echo ""
echo -e "${GREEN}=== Installation Complete! ===${NC}"
echo -e "${BLUE}Access your server at: http://${SERVER_IP}${NC}"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "  cd $INSTALL_DIR"
echo "  docker compose logs -f     # View logs"
echo "  docker compose restart     # Restart services"
echo "  docker compose down        # Stop services"
echo "  bash scripts/setup-auth.sh # Add more users"
echo ""
echo -e "${YELLOW}To enable HTTPS:${NC}"
echo "  1. Place SSL certificates in nginx/ssl/"
echo "  2. Uncomment SSL lines in nginx/nginx.conf"
echo "  3. Run: docker compose restart nginx"

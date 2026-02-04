#!/bin/bash
# Deployment script for SoraWatermarkCleaner

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== SoraWM Deployment Script ===${NC}"

cd "$PROJECT_DIR"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Check if NVIDIA Container Toolkit is installed (for GPU support)
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi not found. GPU support may not be available.${NC}"
fi

# Check if .htpasswd exists
if [ ! -f "$PROJECT_DIR/nginx/.htpasswd" ]; then
    echo -e "${YELLOW}No .htpasswd file found. Setting up authentication...${NC}"
    bash "$SCRIPT_DIR/setup-auth.sh"
fi

# Parse arguments
ACTION=${1:-"up"}

case $ACTION in
    "build")
        echo -e "${GREEN}Building Docker images...${NC}"
        docker compose build --no-cache
        ;;
    "up")
        echo -e "${GREEN}Starting services...${NC}"
        docker compose up -d
        echo -e "${GREEN}Services started!${NC}"
        echo -e "${BLUE}Access the application at: http://$(hostname -I | awk '{print $1}')${NC}"
        ;;
    "down")
        echo -e "${YELLOW}Stopping services...${NC}"
        docker compose down
        ;;
    "restart")
        echo -e "${YELLOW}Restarting services...${NC}"
        docker compose restart
        ;;
    "logs")
        docker compose logs -f
        ;;
    "status")
        docker compose ps
        ;;
    "pull")
        echo -e "${GREEN}Pulling latest changes...${NC}"
        git pull
        docker compose build
        docker compose up -d
        ;;
    *)
        echo "Usage: $0 {build|up|down|restart|logs|status|pull}"
        exit 1
        ;;
esac

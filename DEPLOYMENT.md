# SoraWM Deployment Guide

## Quick Start (GPU Server)

### One-Click Installation

On a fresh Ubuntu 22.04 server with NVIDIA GPU (DigitalOcean GPU Droplet, etc.):

```bash
# Download and run the installation script
curl -fsSL https://raw.githubusercontent.com/linkedlist771/SoraWatermarkCleaner/main/scripts/install-gpu-server.sh | sudo bash
```

Or manually:

```bash
git clone https://github.com/linkedlist771/SoraWatermarkCleaner.git
cd SoraWatermarkCleaner
sudo bash scripts/install-gpu-server.sh
```

## Manual Deployment

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

### Step 1: Clone Repository

```bash
git clone https://github.com/linkedlist771/SoraWatermarkCleaner.git
cd SoraWatermarkCleaner
```

### Step 2: Setup Authentication

```bash
bash scripts/setup-auth.sh
```

This will prompt you to create a username and password for web access.

### Step 3: Build and Start

```bash
# Build Docker images
docker compose build

# Start services
docker compose up -d
```

### Step 4: Access

Open `http://YOUR_SERVER_IP` in your browser. You'll be prompted for the username and password you created.

## Configuration

### Adding More Users

```bash
bash scripts/setup-auth.sh
```

### Enabling HTTPS

1. Obtain SSL certificates (e.g., from Let's Encrypt)

2. Place certificates in `nginx/ssl/`:
   - `fullchain.pem` - Full certificate chain
   - `privkey.pem` - Private key

3. Edit `nginx/nginx.conf`:
   - Uncomment the SSL server block
   - Uncomment the HTTP to HTTPS redirect

4. Restart nginx:
   ```bash
   docker compose restart nginx
   ```

### Let's Encrypt (Certbot)

```bash
# Install certbot
apt-get install certbot

# Get certificate (stop nginx first)
docker compose stop nginx
certbot certonly --standalone -d your-domain.com

# Copy certificates
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/
cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/

# Restart
docker compose up -d
```

## Management Commands

```bash
# View logs
docker compose logs -f

# View app logs only
docker compose logs -f app

# Restart services
docker compose restart

# Stop services
docker compose down

# Rebuild after code changes
docker compose build && docker compose up -d

# Check status
docker compose ps
```

## Recommended GPU Instances

| Provider | GPU | VRAM | Price/hr | Notes |
|----------|-----|------|----------|-------|
| DigitalOcean | A10 | 24GB | ~$1.50 | Good balance |
| DigitalOcean | A100 | 40GB | ~$2.50 | High performance |
| Lambda Labs | A10 | 24GB | ~$0.75 | Budget option |
| Paperspace | A100 | 40GB | ~$3.00 | Easy setup |

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

### Container Won't Start

```bash
# Check logs
docker compose logs app

# Rebuild
docker compose build --no-cache
docker compose up -d
```

### Authentication Issues

```bash
# Recreate .htpasswd
rm nginx/.htpasswd
bash scripts/setup-auth.sh
docker compose restart nginx
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│    Nginx    │────▶│   FastAPI   │
│  (Browser)  │     │  (Auth/SSL) │     │    (App)    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                    │
                           │                    ▼
                           │            ┌─────────────┐
                           │            │  GPU/CUDA   │
                           │            │  (E2FGVI)   │
                           │            └─────────────┘
                           ▼
                    ┌─────────────┐
                    │   Volumes   │
                    │ (checkpoints│
                    │  uploads)   │
                    └─────────────┘
```

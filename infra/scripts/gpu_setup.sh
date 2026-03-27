#!/usr/bin/env bash
# gpu_setup.sh
# ------------
# Sets up the full data-flywheel stack on any Ubuntu instance with an NVIDIA GPU.
# Tested on Lambda Labs, Vast.ai, and RunPod.
#
# Usage:
#   bash infra/scripts/gpu_setup.sh

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════╗"
echo "║   data-flywheel — GPU Setup              ║"
echo "╚══════════════════════════════════════════╝"
echo -e "${NC}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Verify GPU ────────────────────────────────────────────────────────────
echo -e "${YELLOW}Checking GPU...${NC}"
if ! nvidia-smi &>/dev/null; then
    echo -e "${RED}No GPU detected. Make sure you are on a GPU instance:${NC}"
    echo "  Lambda Labs → select a GPU instance type"
    echo "  Vast.ai     → filter by GPU when renting"
    echo "  RunPod      → select a GPU pod"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo -e "${GREEN}✓ GPU available${NC}"

# ── Install Docker ────────────────────────────────────────────────────────
echo -e "\n${YELLOW}Installing Docker...${NC}"
if command -v docker &>/dev/null; then
    echo -e "${GREEN}✓ Docker already installed${NC}"
else
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$(whoami)"
    sudo systemctl enable --now docker
    echo -e "${GREEN}✓ Docker installed${NC}"
fi

# ── Install NVIDIA Container Toolkit ─────────────────────────────────────
echo -e "\n${YELLOW}Installing NVIDIA Container Toolkit...${NC}"
if dpkg -l | grep -q nvidia-container-toolkit 2>/dev/null; then
    echo -e "${GREEN}✓ NVIDIA Container Toolkit already installed${NC}"
else
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update -qq
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo -e "${GREEN}✓ NVIDIA Container Toolkit installed${NC}"
fi

# ── Verify Docker can see GPU ─────────────────────────────────────────────
echo -e "\n${YELLOW}Verifying Docker GPU access...${NC}"
sudo docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi \
    && echo -e "${GREEN}✓ Docker can access GPU${NC}" \
    || { echo -e "${RED}Docker cannot access GPU — check toolkit install${NC}"; exit 1; }

# ── .env check ────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}Checking environment...${NC}"
if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
    if [[ -f "$PROJECT_ROOT/.env.sample" ]]; then
        cp "$PROJECT_ROOT/.env.sample" "$PROJECT_ROOT/.env"
        echo -e "${YELLOW}⚠  .env created from .env.sample — make sure your API keys are set${NC}"
    else
        echo -e "${RED}No .env file found. Copy .env.sample to .env and add your API keys.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ .env found${NC}"
fi

# Check required keys are set
set -a; source "$PROJECT_ROOT/.env" > /dev/null 2>&1; set +a
MISSING=()
[[ -z "${GROQ_API_KEY:-}" || "$GROQ_API_KEY" == "your_groq_api_key_here" ]] && MISSING+=("GROQ_API_KEY")
[[ -z "${HF_TOKEN:-}" || "$HF_TOKEN" == "your_huggingface_token_here" ]] && MISSING+=("HF_TOKEN")
[[ -z "${HF_USERNAME:-}" || "$HF_USERNAME" == "your_huggingface_username_here" ]] && MISSING+=("HF_USERNAME")

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo -e "${RED}Missing required API keys in .env: ${MISSING[*]}${NC}"
    echo "Update .env and re-run this script."
    exit 1
fi
echo -e "${GREEN}✓ API keys present${NC}"

# ── Install Python deps ───────────────────────────────────────────────────
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt -r requirements-dev.txt -q
echo -e "${GREEN}✓ Dependencies installed${NC}"

# ── Start services ────────────────────────────────────────────────────────
echo -e "\n${YELLOW}Starting services...${NC}"
sudo docker compose -f infra/docker/docker-compose.yml up -d --build
echo -e "${GREEN}✓ Services started${NC}"

# ── Wait for API ──────────────────────────────────────────────────────────
echo -e "\n${YELLOW}Waiting for API to be healthy...${NC}"
MAX_WAIT=90
ELAPSED=0
while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    if curl -sf http://localhost:8000/health &>/dev/null; then
        echo -e "${GREEN}✓ API is up${NC}"
        break
    fi
    sleep 3
    ELAPSED=$((ELAPSED + 3))
    echo -n "."
done

if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    echo -e "\n${RED}Timed out waiting for API. Check logs:${NC}"
    echo "  sudo docker logs flywheel-orchestrator"
    exit 1
fi

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗"
echo -e "║        GPU setup complete! ✓             ║"
echo -e "╚══════════════════════════════════════════╝${NC}"
echo ""
echo "Services running:"
echo "  API    → http://localhost:8000"
echo "  MLflow → http://localhost:5000"
echo ""
echo "Next steps:"
echo "  make seed          # seed inference logs"
echo "  make run-flywheel  # trigger a flywheel cycle (GPU LoRA SFT enabled)"
echo "  make logs          # tail worker logs"
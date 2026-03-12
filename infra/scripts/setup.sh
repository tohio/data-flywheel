#!/usr/bin/env bash
# setup.sh
# --------
# First-time setup for the data-flywheel project.
# Checks prerequisites, creates .env from template,
# pulls Docker images, and verifies the stack starts cleanly.
#
# Usage:
#   bash infra/scripts/setup.sh

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/infra/docker/docker-compose.yml"

cd "$PROJECT_ROOT"

echo -e "${BLUE}"
echo "╔══════════════════════════════════════╗"
echo "║      data-flywheel — setup           ║"
echo "╚══════════════════════════════════════╝"
echo -e "${NC}"

# ── Check prerequisites ───────────────────────────────────────────────────
echo -e "${YELLOW}Checking prerequisites...${NC}"

check_cmd() {
    if command -v "$1" &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $1"
    else
        echo -e "  ${RED}✗ $1 not found — please install it first${NC}"
        exit 1
    fi
}

check_cmd docker
check_cmd python3

# Docker Compose (v2 plugin or standalone)
if docker compose version &>/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} docker compose (v2)"
elif command -v docker-compose &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} docker-compose (v1)"
else
    echo -e "  ${RED}✗ docker compose not found${NC}"
    exit 1
fi

# Python version
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 12 ]]; then
    echo -e "  ${GREEN}✓${NC} python $PY_VERSION"
else
    echo -e "  ${YELLOW}⚠${NC}  python $PY_VERSION (3.12+ recommended)"
fi

# ── .env setup ────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Configuring environment...${NC}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    echo -e "  ${GREEN}✓${NC} .env already exists — skipping"
else
    cp "$PROJECT_ROOT/.env.sample" "$PROJECT_ROOT/.env"
    echo -e "  ${GREEN}✓${NC} .env created from .env.sample"
    echo ""
    echo -e "  ${YELLOW}⚠  Action required: add your API keys to .env${NC}"
    echo "     GROQ_API_KEY  → https://console.groq.com"
    echo "     HF_TOKEN      → https://huggingface.co/settings/tokens"
fi

# Check for required API keys
source "$PROJECT_ROOT/.env" 2>/dev/null || true
if [[ -z "${GROQ_API_KEY:-}" ]] || [[ "$GROQ_API_KEY" == "your_groq_api_key_here" ]]; then
    echo -e "  ${YELLOW}⚠  GROQ_API_KEY not set in .env${NC}"
fi
if [[ -z "${HF_TOKEN:-}" ]] || [[ "$HF_TOKEN" == "your_huggingface_token_here" ]]; then
    echo -e "  ${YELLOW}⚠  HF_TOKEN not set in .env${NC}"
fi

# ── Install Python deps ───────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt -r requirements-dev.txt -q
echo -e "  ${GREEN}✓${NC} dependencies installed"

# ── Pull Docker images ────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Pulling Docker images...${NC}"
docker compose -f "$COMPOSE_FILE" pull --quiet
echo -e "  ${GREEN}✓${NC} images pulled"

# ── Start services ────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Starting services...${NC}"
docker compose -f "$COMPOSE_FILE" up -d --build
echo -e "  ${GREEN}✓${NC} services started"

# ── Wait for health ───────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
MAX_WAIT=60
ELAPSED=0
while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    if curl -sf http://localhost:8000/health &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} orchestrator API is up"
        break
    fi
    sleep 3
    ELAPSED=$((ELAPSED + 3))
    echo -n "."
done

if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    echo -e "\n  ${RED}✗ Timed out waiting for API — check logs: make logs${NC}"
    exit 1
fi

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════╗"
echo -e "║        Setup complete! ✓             ║"
echo -e "╚══════════════════════════════════════╝${NC}"
echo ""
echo "Services running:"
echo "  API         → http://localhost:8000"
echo "  API docs    → http://localhost:8000/docs"
echo "  MLflow      → http://localhost:5000"
echo "  MongoDB     → localhost:27017"
echo "  Redis       → localhost:6379"
echo "  Elasticsearch → http://localhost:9200"
echo ""
echo "Next steps:"
echo "  make seed          # seed synthetic inference logs"
echo "  make run-flywheel  # trigger a flywheel cycle"
echo "  make logs          # tail orchestrator logs"
echo "  pytest tests/unit/ # run unit tests (no services needed)"

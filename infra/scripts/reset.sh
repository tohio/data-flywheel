#!/usr/bin/env bash
# reset.sh
# --------
# Wipes all local state — databases, logs, MLflow artifacts, Docker volumes.
# Useful for a clean re-run during development.
#
# Usage:
#   bash infra/scripts/reset.sh           # prompts for confirmation
#   bash infra/scripts/reset.sh --force   # no prompt

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

COMPOSE_FILE="infra/docker/docker-compose.yml"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$PROJECT_ROOT"

# ── Confirmation ──────────────────────────────────────────────────────────
if [[ "${1:-}" != "--force" ]]; then
    echo -e "${YELLOW}⚠️  This will destroy all local data:${NC}"
    echo "   • MongoDB (flywheel_runs, experiments, datasets, model_registry)"
    echo "   • Elasticsearch (inference_logs, curated_datasets)"
    echo "   • Redis (Celery queues and results)"
    echo "   • MLflow artifacts"
    echo "   • All Docker volumes"
    echo ""
    read -rp "Are you sure? (yes/N): " confirm
    if [[ "$confirm" != "yes" ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo ""
echo "Resetting data-flywheel local environment..."

# ── Stop services ─────────────────────────────────────────────────────────
echo -e "\n${YELLOW}→ Stopping services...${NC}"
docker compose -f "$COMPOSE_FILE" down --volumes --remove-orphans 2>/dev/null || true

# ── Remove named volumes ──────────────────────────────────────────────────
echo -e "${YELLOW}→ Removing Docker volumes...${NC}"
volumes=(
    "data-flywheel_mongodb_data"
    "data-flywheel_redis_data"
    "data-flywheel_elasticsearch_data"
    "data-flywheel_mlflow_artifacts"
)
for vol in "${volumes[@]}"; do
    docker volume rm "$vol" 2>/dev/null && echo "  removed: $vol" || echo "  skipped: $vol (not found)"
done

# ── Clear MLflow local artifacts ──────────────────────────────────────────
echo -e "${YELLOW}→ Clearing MLflow artifacts...${NC}"
rm -rf "$PROJECT_ROOT/experiments/"* 2>/dev/null || true
rm -rf "$PROJECT_ROOT/mlruns/" 2>/dev/null || true
echo "  cleared: experiments/ and mlruns/"

# ── Clear pytest cache ────────────────────────────────────────────────────
echo -e "${YELLOW}→ Clearing test cache...${NC}"
find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_ROOT" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true
echo "  cleared: __pycache__, .pytest_cache, *.pyc"

echo ""
echo -e "${GREEN}✓ Reset complete.${NC}"
echo ""
echo "To start fresh:"
echo "  make up       # start all services"
echo "  make seed     # seed inference logs"
echo "  make run-flywheel  # trigger a flywheel cycle"

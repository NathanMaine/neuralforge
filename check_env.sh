#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# NeuralForge Preflight Check
# Validates system requirements before deployment.
#
# Usage:  bash check_env.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

PASS=0
WARN=0
FAIL=0

pass()  { echo -e "  ${GREEN}[PASS]${NC} $1"; ((PASS++)); }
warn()  { echo -e "  ${YELLOW}[WARN]${NC} $1"; ((WARN++)); }
fail()  { echo -e "  ${RED}[FAIL]${NC} $1"; ((FAIL++)); }

echo ""
echo -e "${BOLD}${CYAN}  NeuralForge Preflight Check${NC}"
echo -e "${CYAN}  ─────────────────────────────────────────${NC}"
echo ""

# ── 1. NVIDIA GPU ─────────────────────────────────────────
echo -e "${BOLD}GPU${NC}"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_NAME" ]; then
        pass "NVIDIA GPU detected: $GPU_NAME"
    else
        fail "nvidia-smi found but no GPU detected"
    fi
else
    fail "nvidia-smi not found — NVIDIA driver not installed"
fi

# ── 2. VRAM ───────────────────────────────────────────────
echo -e "${BOLD}VRAM${NC}"
if command -v nvidia-smi &>/dev/null; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$VRAM_MB" ]; then
        VRAM_GB=$((VRAM_MB / 1024))
        if [ "$VRAM_GB" -ge 16 ]; then
            pass "VRAM: ${VRAM_GB}GB (minimum 16GB)"
        elif [ "$VRAM_GB" -ge 8 ]; then
            warn "VRAM: ${VRAM_GB}GB — 16GB+ recommended for full stack"
        else
            fail "VRAM: ${VRAM_GB}GB — minimum 16GB required"
        fi
    else
        warn "Could not query VRAM"
    fi
else
    fail "Cannot check VRAM — nvidia-smi not available"
fi

# ── 3. CUDA Version ──────────────────────────────────────
echo -e "${BOLD}CUDA${NC}"
if command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$CUDA_VERSION" ]; then
        pass "NVIDIA Driver: $CUDA_VERSION"
    else
        warn "Could not determine driver version"
    fi
else
    fail "Cannot check CUDA — nvidia-smi not available"
fi

# ── 4. Docker ─────────────────────────────────────────────
echo -e "${BOLD}Docker${NC}"
if command -v docker &>/dev/null; then
    DOCKER_VERSION=$(docker --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1)
    if [ -n "$DOCKER_VERSION" ]; then
        pass "Docker: v$DOCKER_VERSION"
    else
        pass "Docker installed"
    fi

    # Check Docker daemon
    if docker info &>/dev/null; then
        pass "Docker daemon running"
    else
        fail "Docker daemon not running — start with: sudo systemctl start docker"
    fi
else
    fail "Docker not installed — https://docs.docker.com/engine/install/"
fi

# ── 5. Docker Compose ────────────────────────────────────
echo -e "${BOLD}Docker Compose${NC}"
if docker compose version &>/dev/null; then
    COMPOSE_VERSION=$(docker compose version --short 2>/dev/null)
    pass "Docker Compose: v$COMPOSE_VERSION"
elif command -v docker-compose &>/dev/null; then
    warn "Legacy docker-compose found — upgrade to Docker Compose V2"
else
    fail "Docker Compose not installed"
fi

# ── 6. NVIDIA Container Toolkit ──────────────────────────
echo -e "${BOLD}NVIDIA Container Toolkit${NC}"
if command -v nvidia-container-cli &>/dev/null; then
    pass "NVIDIA Container Toolkit installed"
elif docker info 2>/dev/null | grep -qi nvidia; then
    pass "NVIDIA runtime detected in Docker"
else
    # Try running a quick GPU container test
    if docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi &>/dev/null 2>&1; then
        pass "GPU containers working"
    else
        fail "NVIDIA Container Toolkit not detected — https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
fi

# ── 7. NGC API Key ────────────────────────────────────────
echo -e "${BOLD}NGC API Key${NC}"
if [ -n "${NGC_API_KEY:-}" ]; then
    # Mask the key for display
    KEY_LEN=${#NGC_API_KEY}
    if [ "$KEY_LEN" -gt 8 ]; then
        MASKED="${NGC_API_KEY:0:4}...${NGC_API_KEY: -4}"
        pass "NGC_API_KEY set ($MASKED)"
    else
        warn "NGC_API_KEY looks too short"
    fi
elif [ -f ".env" ] && grep -q "NGC_API_KEY" .env 2>/dev/null; then
    KEY_VALUE=$(grep "NGC_API_KEY" .env | head -1 | cut -d= -f2-)
    if [ -n "$KEY_VALUE" ] && [ "$KEY_VALUE" != "your-ngc-api-key-here" ]; then
        pass "NGC_API_KEY found in .env"
    else
        fail "NGC_API_KEY in .env is not set — get yours at https://org.ngc.nvidia.com/setup"
    fi
else
    fail "NGC_API_KEY not set — export NGC_API_KEY=... or add to .env"
fi

# ── 8. Disk Space ─────────────────────────────────────────
echo -e "${BOLD}Disk Space${NC}"
AVAIL_KB=$(df -k . 2>/dev/null | tail -1 | awk '{print $4}')
if [ -n "$AVAIL_KB" ]; then
    AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
    if [ "$AVAIL_GB" -ge 50 ]; then
        pass "Available disk: ${AVAIL_GB}GB (minimum 50GB)"
    elif [ "$AVAIL_GB" -ge 20 ]; then
        warn "Available disk: ${AVAIL_GB}GB — 50GB+ recommended for NIM model cache"
    else
        fail "Available disk: ${AVAIL_GB}GB — minimum 50GB required"
    fi
else
    warn "Could not determine available disk space"
fi

# ── 9. Port Availability ─────────────────────────────────
echo -e "${BOLD}Ports${NC}"
PORTS=(8090 8000 8001 6333)
PORT_NAMES=("NeuralForge API" "NIM LLM" "Triton HTTP" "Qdrant")
for i in "${!PORTS[@]}"; do
    PORT=${PORTS[$i]}
    NAME=${PORT_NAMES[$i]}
    if ! (echo >/dev/tcp/localhost/$PORT) 2>/dev/null; then
        pass "Port $PORT available ($NAME)"
    else
        warn "Port $PORT in use — $NAME may conflict"
    fi
done

# ── Summary ───────────────────────────────────────────────
echo ""
echo -e "${CYAN}  ─────────────────────────────────────────${NC}"
TOTAL=$((PASS + WARN + FAIL))
echo -e "  ${GREEN}$PASS passed${NC}  ${YELLOW}$WARN warnings${NC}  ${RED}$FAIL failed${NC}  (${TOTAL} checks)"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "  ${RED}${BOLD}Fix failures above before deploying.${NC}"
    echo ""
    exit 1
elif [ "$WARN" -gt 0 ]; then
    echo -e "  ${YELLOW}${BOLD}Ready with warnings — review items above.${NC}"
    echo ""
    exit 0
else
    echo -e "  ${GREEN}${BOLD}All checks passed — ready to deploy!${NC}"
    echo -e "  Run: ${CYAN}docker compose up -d${NC}"
    echo ""
    exit 0
fi

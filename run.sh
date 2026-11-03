#!/usr/bin/env bash
# run.sh — Khởi chạy Dashboard Customer Churn Retention
#
# Dùng cho nhóm: bash run.sh
# Bật live API: bash run.sh --live
# Chỉ cài môi trường: bash run.sh --setup-only
# Xem help: bash run.sh --help
#
# Yêu cầu: Python 3.12–3.14 đã cài sẵn.
# Nền tảng đã kiểm chứng: macOS (Intel/Apple Silicon), Ubuntu/Debian.
# Windows: chạy qua WSL (Windows Subsystem for Linux).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── Màu sắc terminal ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[run.sh]${NC} $*"; }
success() { echo -e "${GREEN}[run.sh] ✓${NC} $*"; }
warn()    { echo -e "${YELLOW}[run.sh] ⚠${NC} $*"; }
error()   { echo -e "${RED}[run.sh] ✗${NC} $*" >&2; }
die()     { error "$*"; exit 1; }

# ── Parse arguments ───────────────────────────────────────────────────────────
ENABLE_LIVE=false
SETUP_ONLY=false
PORT=8501
STATE_DIR=""

usage() {
    echo ""
    echo -e "${BOLD}Cách dùng:${NC}"
    echo "  bash run.sh [tuỳ chọn]"
    echo ""
    echo -e "${BOLD}Tuỳ chọn:${NC}"
    echo "  --live            Bật LLM live API (gọi Groq/Gemini thật, tốn quota)"
    echo "  --setup-only      Chỉ cài môi trường, không khởi động Dashboard"
    echo "  --port PORT       Cổng Streamlit (mặc định: 8501)"
    echo "  --state-dir DIR   Thư mục lưu trạng thái demo (mặc định: tạo tạm)"
    echo "  --help            Hiển thị hướng dẫn này"
    echo ""
    echo -e "${BOLD}Ví dụ:${NC}"
    echo "  bash run.sh                     # Chế độ mock (không tốn quota)"
    echo "  bash run.sh --live              # Chế độ live LLM thật"
    echo "  bash run.sh --port 8502 --live  # Live trên cổng khác"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --live)        ENABLE_LIVE=true;  shift ;;
        --setup-only)  SETUP_ONLY=true;   shift ;;
        --port)        PORT="$2";          shift 2 ;;
        --state-dir)   STATE_DIR="$2";     shift 2 ;;
        --help|-h)     usage; exit 0 ;;
        *) die "Tham số không hợp lệ: $1. Dùng --help để xem hướng dẫn." ;;
    esac
done

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  Customer Churn Retention Dashboard${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ── Bước 1: Kiểm tra Python ───────────────────────────────────────────────────
info "Bước 1/6: Kiểm tra Python runtime..."

PYTHON=""
for cmd in python3.14 python3.13 python3.12 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        MAJOR="${VER%%.*}"
        MINOR="${VER##*.}"
        if [[ "$MAJOR" == "3" ]] && [[ "$MINOR" -ge 12 ]] && [[ "$MINOR" -le 14 ]]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    error "Python 3.12, 3.13 hoặc 3.14 không tìm thấy."
    echo ""
    echo -e "  ${BOLD}Cách cài Python:${NC}"
    echo "  • macOS:   brew install python@3.12"
    echo "             hoặc tải từ https://www.python.org/downloads/"
    echo "  • Linux:   sudo apt install python3.12 python3.12-venv"
    echo "  • Windows: cài WSL rồi: sudo apt install python3.12 python3.12-venv"
    echo ""
    exit 1
fi

PYTHON_VERSION=$("$PYTHON" -c "import sys; print(sys.version)" 2>/dev/null)
success "Dùng Python: $PYTHON  ($PYTHON_VERSION)"

# ── Bước 2: Tạo / kiểm tra virtualenv ────────────────────────────────────────
info "Bước 2/6: Chuẩn bị môi trường ảo..."

VENV_DIR="$REPO_ROOT/.venv"

if [[ -d "$VENV_DIR" ]]; then
    # Kiểm tra virtualenv còn dùng được không
    VENV_PYTHON="$VENV_DIR/bin/python3"
    if [[ ! -x "$VENV_PYTHON" ]]; then
        warn "Virtualenv bị hỏng, tạo lại..."
        rm -rf "$VENV_DIR"
    else
        VENV_VER=$("$VENV_PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        VENV_MAJOR="${VENV_VER%%.*}"
        VENV_MINOR="${VENV_VER##*.}"
        if [[ "$VENV_MAJOR" != "3" ]] || [[ "$VENV_MINOR" -lt 12 ]]; then
            warn "Virtualenv dùng Python $VENV_VER cũ hơn yêu cầu, tạo lại..."
            rm -rf "$VENV_DIR"
        fi
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    info "  Tạo virtualenv mới tại $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR" || die "Không tạo được virtualenv. Thử: $PYTHON -m pip install virtualenv"
fi

VENV_PYTHON="$VENV_DIR/bin/python3"
VENV_PIP="$VENV_DIR/bin/pip"
success "Virtualenv sẵn sàng: $VENV_DIR"

# ── Bước 3: Cài dependency ────────────────────────────────────────────────────
info "Bước 3/6: Cài dependency (theo constraints/rag-v2.txt)..."

"$VENV_PIP" install --quiet --upgrade pip 2>&1 | tail -1 || true
"$VENV_PIP" install --quiet -c constraints/rag-v2.txt -e ".[dev]" \
    || die "Cài dependency thất bại. Kiểm tra kết nối mạng và thử lại."

success "Dependency đã được cài đặt."

# ── Bước 4: Nạp cấu hình ──────────────────────────────────────────────────────
info "Bước 4/6: Nạp cấu hình..."

# Nạp .env.shared (cấu hình dùng chung của nhóm)
if [[ -f "$REPO_ROOT/.env.shared" ]]; then
    set -a
    # shellcheck source=.env.shared
    source "$REPO_ROOT/.env.shared"
    set +a
    success "Đã nạp .env.shared (cấu hình nhóm)."
else
    warn ".env.shared không tìm thấy — cần có file này để dùng LLM thật."
fi

# Cho phép .env.local ghi đè không bắt buộc (cấu hình riêng máy)
if [[ -f "$REPO_ROOT/.env.local" ]]; then
    set -a
    # shellcheck source=.env.local
    source "$REPO_ROOT/.env.local"
    set +a
    success "Đã nạp .env.local (ghi đè cấu hình cá nhân)."
fi

# ── Bước 5: Kiểm tra artifact RAG ────────────────────────────────────────────
info "Bước 5/6: Kiểm tra artifact RAG..."

ARTIFACT="${CHURN_LLM_REFERENCE_ARTIFACT:-artifacts/references/telco-train-v2.zip}"
EXPECTED_SHA="${CHURN_LLM_REFERENCE_SHA256:-c85e61be7edfd0e91b248e0e353088677a345a378e64840b1613b51194b40bf8}"

if [[ ! -f "$REPO_ROOT/$ARTIFACT" ]]; then
    die "Artifact không tìm thấy: $ARTIFACT\n  Thử: git pull (artifact phải có trong repo)."
fi

# Kiểm tra SHA256
if command -v sha256sum &>/dev/null; then
    ACTUAL_SHA=$(sha256sum "$REPO_ROOT/$ARTIFACT" | cut -d' ' -f1)
elif command -v shasum &>/dev/null; then
    ACTUAL_SHA=$(shasum -a 256 "$REPO_ROOT/$ARTIFACT" | cut -d' ' -f1)
else
    warn "Không thể kiểm tra checksum (sha256sum/shasum không có). Bỏ qua."
    ACTUAL_SHA="$EXPECTED_SHA"
fi

if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
    die "Artifact bị hỏng hoặc sai phiên bản!\n  Thực tế:  $ACTUAL_SHA\n  Yêu cầu: $EXPECTED_SHA\n  Thử: git checkout HEAD -- $ARTIFACT"
fi

# Đặt biến môi trường cho app nếu chưa có
export CHURN_LLM_REFERENCE_ARTIFACT="$REPO_ROOT/$ARTIFACT"
export CHURN_LLM_REFERENCE_SHA256="$EXPECTED_SHA"

success "Artifact hợp lệ: $ARTIFACT"

# ── Kiểm tra LLM keys (chỉ cần khi live) ─────────────────────────────────────
if [[ "$ENABLE_LIVE" == "true" ]]; then
    MISSING_KEYS=()
    [[ -z "${GROQ_API_KEY:-}" ]]   && MISSING_KEYS+=("GROQ_API_KEY")
    [[ -z "${GEMINI_API_KEY:-}" ]] && MISSING_KEYS+=("GEMINI_API_KEY")
    if [[ ${#MISSING_KEYS[@]} -gt 0 ]]; then
        die "Thiếu API key để dùng chế độ live: ${MISSING_KEYS[*]}\n  Kiểm tra .env.shared hoặc đặt biến môi trường thủ công."
    fi
    export CHURN_DASHBOARD_ALLOW_LIVE=1
    export CHURN_DASHBOARD_PREVIEW=0
else
    # Chế độ mock: xoá keys khỏi env để tránh gọi nhầm
    unset GROQ_API_KEY GEMINI_API_KEY GOOGLE_API_KEY OPENAI_API_KEY 2>/dev/null || true
    export CHURN_DASHBOARD_ALLOW_LIVE=0
    export CHURN_DASHBOARD_PREVIEW=1
fi

# Nếu chỉ setup thì dừng ở đây
if [[ "$SETUP_ONLY" == "true" ]]; then
    success "Cài đặt hoàn tất. Chạy lại không có --setup-only để khởi động Dashboard."
    exit 0
fi

# ── Bước 6: Khởi động Dashboard ───────────────────────────────────────────────
info "Bước 6/6: Khởi động Dashboard..."

if [[ "$ENABLE_LIVE" == "true" ]]; then
    echo ""
    warn "Chế độ LIVE: mỗi lần phân tích sẽ gọi Groq/Gemini API (tốn quota free)."
    warn "Xác nhận bằng cách nhập ENABLE LIVE API khi được hỏi."
    echo ""
fi

MODE_STR="mock"
[[ "$ENABLE_LIVE" == "true" ]] && MODE_STR="live"

echo ""
echo -e "${GREEN}${BOLD}  Dashboard sắp khởi động tại: http://127.0.0.1:${PORT}${NC}"
echo -e "  Chế độ: ${BOLD}${MODE_STR}${NC}"
echo -e "  Dừng:   ${BOLD}Ctrl+C${NC}"
echo ""

export PYTHONPATH="$REPO_ROOT/src"

EXTRA_ARGS=()
[[ -n "$STATE_DIR" ]] && EXTRA_ARGS+=("--state-dir" "$STATE_DIR")
[[ "$ENABLE_LIVE" == "true" ]] && EXTRA_ARGS+=("--enable-live")

exec "$VENV_PYTHON" -m churn_retention.dashboard.demo \
    --port "$PORT" \
    "${EXTRA_ARGS[@]}"

#!/usr/bin/env bash
# run.sh — Khởi chạy Dashboard Customer Churn Retention
#
# Dành cho thành viên nhóm (repo private):
#   bash run.sh
#
# Các tuỳ chọn bổ sung:
#   bash run.sh --mock        # Chạy chế độ mô phỏng (không gọi API ngoài)
#   bash run.sh --setup-only  # Chỉ cài môi trường & migrations, không mở web
#   bash run.sh --port 8502   # Mở trên cổng khác (mặc định: 8501)
#   bash run.sh --help        # Xem hướng dẫn chi tiết
#
# Yêu cầu cài sẵn: Python 3.12, 3.13 hoặc 3.14.
# Nền tảng đã kiểm chứng: macOS (Apple Silicon / Intel), Linux (Ubuntu/Debian).
# Windows: chạy thông qua WSL (Windows Subsystem for Linux).

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

# ── Tham số dòng lệnh ─────────────────────────────────────────────────────────
ENABLE_LIVE=true
SETUP_ONLY=false
PORT=8501
STATE_DIR="$REPO_ROOT/data/dashboard_state"

usage() {
    echo ""
    echo -e "${BOLD}Cách dùng:${NC}"
    echo "  bash run.sh [tuỳ chọn]"
    echo ""
    echo -e "${BOLD}Mặc định:${NC}"
    echo "  Khởi chạy Dashboard với luồng LLM thật (Groq/Gemini dùng key nhóm trong .env.shared)."
    echo ""
    echo -e "${BOLD}Tuỳ chọn:${NC}"
    echo "  --mock            Chạy chế độ mô phỏng (offline demo, không gọi API)"
    echo "  --live            Bật chế độ LLM thật (mặc định đã bật)"
    echo "  --setup-only      Chỉ cài môi trường, migrations và kiểm tra artifact, không mở web"
    echo "  --port PORT       Cổng Streamlit (mặc định: 8501)"
    echo "  --state-dir DIR   Thư mục lưu trạng thái SQLite (mặc định: thư mục tạm 0700 an toàn)"
    echo "  --help, -h        Hiển thị hướng dẫn này"
    echo ""
    echo -e "${BOLD}Ví dụ:${NC}"
    echo "  bash run.sh                   # Chạy trực tiếp với LLM thật"
    echo "  bash run.sh --mock            # Chạy offline mock"
    echo "  bash run.sh --port 8502       # Chạy trên cổng 8502"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mock)        ENABLE_LIVE=false; shift ;;
        --live)        ENABLE_LIVE=true;  shift ;;
        --setup-only)  SETUP_ONLY=true;   shift ;;
        --port)        PORT="$2";         shift 2 ;;
        --state-dir)   STATE_DIR="$2";    shift 2 ;;
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
    error "Python 3.12, 3.13 hoặc 3.14 không tìm thấy trên hệ thống."
    echo ""
    echo -e "  ${BOLD}Cách cài đặt Python:${NC}"
    echo "  • macOS:   brew install python@3.12"
    echo "             hoặc tải từ https://www.python.org/downloads/"
    echo "  • Linux:   sudo apt install python3.12 python3.12-venv"
    echo "  • Windows: cài WSL (Windows Subsystem for Linux), sau đó chạy:"
    echo "             sudo apt install python3.12 python3.12-venv"
    echo ""
    exit 1
fi

PYTHON_VERSION=$("$PYTHON" -c "import sys; print(sys.version.split()[0])" 2>/dev/null)
success "Phát hiện Python: $PYTHON ($PYTHON_VERSION)"

# ── Bước 2: Tạo / kiểm tra virtualenv ────────────────────────────────────────
info "Bước 2/6: Chuẩn bị môi trường ảo..."

VENV_DIR="$REPO_ROOT/.venv"

if [[ -d "$VENV_DIR" ]]; then
    VENV_PYTHON="$VENV_DIR/bin/python3"
    if [[ ! -x "$VENV_PYTHON" ]]; then
        warn "Virtualenv bị hỏng, tạo lại..."
        rm -rf "$VENV_DIR"
    else
        VENV_VER=$("$VENV_PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        VENV_MAJOR="${VENV_VER%%.*}"
        VENV_MINOR="${VENV_VER##*.}"
        if [[ "$VENV_MAJOR" != "3" ]] || [[ "$VENV_MINOR" -lt 12 ]]; then
            warn "Virtualenv đang dùng Python $VENV_VER (yêu cầu >= 3.12), tạo lại..."
            rm -rf "$VENV_DIR"
        fi
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    info "  Đang tạo virtualenv tại $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR" || die "Không tạo được virtualenv. Thử cài đặt gói python3-venv."
fi

VENV_PYTHON="$VENV_DIR/bin/python3"
VENV_PIP="$VENV_DIR/bin/pip"
success "Môi trường ảo sẵn sàng: $VENV_DIR"

# ── Bước 3: Cài dependency ────────────────────────────────────────────────────
info "Bước 3/6: Kiểm tra và cài đặt dependency (constraints/rag-v2.txt)..."

if "$VENV_PYTHON" -c "import streamlit, sklearn, pandas, pydantic, groq, google.genai, alembic; from churn_retention.dashboard import demo" &>/dev/null; then
    success "Các gói dependency đã có đầy đủ."
else
    info "  Cài đặt các gói theo khóa phiên bản..."
    "$VENV_PIP" install --quiet --upgrade pip 2>&1 | tail -1 || true
    "$VENV_PIP" install --quiet -c constraints/rag-v2.txt -e ".[dev]" \
        || die "Cài dependency thất bại. Kiểm tra kết nối mạng và thử lại."
    success "Đã cài đặt xong dependency."
fi

# ── Bước 4: Nạp cấu hình ──────────────────────────────────────────────────────
info "Bước 4/6: Nạp cấu hình ứng dụng..."

# Nạp .env.shared (cấu hình dùng chung của nhóm)
if [[ -f "$REPO_ROOT/.env.shared" ]]; then
    set -a
    # shellcheck source=.env.shared
    source "$REPO_ROOT/.env.shared"
    set +a
    success "Đã nạp .env.shared (cấu hình nhóm)."
else
    warn ".env.shared không tìm thấy. Nếu cần gọi LLM thật, vui lòng kiểm tra repo private."
fi

# Cho phép .env.local ghi đè cá nhân (không bắt buộc)
if [[ -f "$REPO_ROOT/.env.local" ]]; then
    set -a
    # shellcheck source=.env.local
    source "$REPO_ROOT/.env.local"
    set +a
    success "Đã nạp .env.local (ghi đè cấu hình cá nhân)."
fi

# Thiết lập database URL mặc định nếu chưa cấu hình
export DATABASE_URL="${DATABASE_URL:-sqlite:///$REPO_ROOT/churn_metadata.db}"

# ── Bước 5: Kiểm tra artifact & chạy migrations ───────────────────────────────
info "Bước 5/6: Kiểm tra RAG artifact và cập nhật database..."

ARTIFACT="${CHURN_LLM_REFERENCE_ARTIFACT:-artifacts/references/telco-train-v2.zip}"
EXPECTED_SHA="${CHURN_LLM_REFERENCE_SHA256:-c85e61be7edfd0e91b248e0e353088677a345a378e64840b1613b51194b40bf8}"

if [[ ! -f "$REPO_ROOT/$ARTIFACT" ]]; then
    die "Artifact không tìm thấy: $ARTIFACT\n  Hãy chạy: git checkout HEAD -- $ARTIFACT"
fi

ACTUAL_SHA=$("$VENV_PYTHON" -c "
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], 'rb') as f:
    while chunk := f.read(65536):
        h.update(chunk)
print(h.hexdigest())
" "$REPO_ROOT/$ARTIFACT" 2>/dev/null || echo "")

if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
    die "Artifact bị sai checksum SHA256!\n  Thực tế:  $ACTUAL_SHA\n  Kỳ vọng:  $EXPECTED_SHA\n  Thử: git checkout HEAD -- $ARTIFACT"
fi

export CHURN_LLM_REFERENCE_ARTIFACT="$REPO_ROOT/$ARTIFACT"
export CHURN_LLM_REFERENCE_SHA256="$EXPECTED_SHA"
success "Artifact RAG v2 hợp lệ."

# Chạy Alembic migrations (an toàn cho dữ liệu đã có)
if "$VENV_PYTHON" -m alembic upgrade head &>/dev/null; then
    success "Database migration hoàn tất (schema đồng bộ)."
else
    die "Chạy database migration thất bại. Kiểm tra DATABASE_URL: $DATABASE_URL"
fi

# Kiểm tra LLM keys khi ở chế độ live
if [[ "$ENABLE_LIVE" == "true" ]]; then
    MISSING_KEYS=()
    [[ -z "${GROQ_API_KEY:-}" ]]   && MISSING_KEYS+=("GROQ_API_KEY")
    [[ -z "${GEMINI_API_KEY:-}" ]] && MISSING_KEYS+=("GEMINI_API_KEY")
    if [[ ${#MISSING_KEYS[@]} -gt 0 ]]; then
        die "Thiếu API key cho chế độ live: ${MISSING_KEYS[*]}\n  Kiểm tra .env.shared hoặc đặt biến môi trường."
    fi
    export CHURN_DASHBOARD_ALLOW_LIVE=1
    export CHURN_DASHBOARD_PREVIEW=0
else
    # Chế độ mock: xoá keys khỏi env để tránh gọi nhầm
    unset GROQ_API_KEY GEMINI_API_KEY GOOGLE_API_KEY OPENAI_API_KEY 2>/dev/null || true
    export CHURN_DASHBOARD_ALLOW_LIVE=0
    export CHURN_DASHBOARD_PREVIEW=1
fi

if [[ "$SETUP_ONLY" == "true" ]]; then
    echo ""
    success "Cài đặt & kiểm tra hoàn tất! Để khởi động Dashboard, chạy: bash run.sh"
    exit 0
fi

# ── Bước 6: Khởi động Dashboard ───────────────────────────────────────────────
info "Bước 6/6: Khởi động Dashboard..."

MODE_STR="Live API (Groq + Gemini)"
[[ "$ENABLE_LIVE" != "true" ]] && MODE_STR="Mock provider (Offline)"

echo ""
echo -e "${GREEN}${BOLD}  ✓ Dashboard đang chạy tại: http://127.0.0.1:${PORT}${NC}"
echo -e "  • Chế độ:    ${BOLD}${MODE_STR}${NC}"
echo -e "  • Dừng chạy: ${BOLD}Ctrl+C${NC}"
echo ""

export PYTHONPATH="$REPO_ROOT/src"

EXTRA_ARGS=()
[[ -n "$STATE_DIR" ]] && EXTRA_ARGS+=("--state-dir" "$STATE_DIR")

if [[ "$ENABLE_LIVE" == "true" ]]; then
    exec "$VENV_PYTHON" -m churn_retention.dashboard.demo \
        --port "$PORT" \
        --enable-live \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} <<< "ENABLE LIVE API"
else
    exec "$VENV_PYTHON" -m churn_retention.dashboard.demo \
        --port "$PORT" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
fi

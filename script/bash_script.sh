#!/usr/bin/env bash
# 1. 가상환경 활성화
#   source .venv/bin/activate
#
# 2. 스크립트에 실행 권한 부여 (최초 1회만)
#   chmod +x ./script/bash_script.sh
#
# 3. 스크립트 실행
#   ./script/bash_script.sh
#
# ========================================================== #
#           자동매매 시스템 통합 제어 스크립트 (EC2용)       #
# ========================================================== #

set -Eeuo pipefail

# ---------------------------------------------------------- #
#                  공통 설정 및 유틸리티 함수                 #
# ---------------------------------------------------------- #

if [[ -f ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

# Streamlit: 온보딩/텔레메트리 비활성 + 헤드리스(프롬프트 방지)
export STREAMLIT_SERVER_HEADLESS="true"
export STREAMLIT_BROWSER_GATHERUSAGESTATS="false"

START_DATE='2018-01-01'
END_DATE='2025-09-30'

invoke_backtest() {
  echo ">> 백테스트 실행: $*"
  "$PY" ./backtest.py "$@"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "!! 백테스트 실패: $*" >&2
    exit $rc
  fi
}

invoke_validate() {
  local run_dir=$1
  echo ">> 검증 실행: ${run_dir}"
  "$PY" ./validate.py "${run_dir}"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "!! 검증 실패: ${run_dir}" >&2
    exit $rc
  fi
}

start_streamlit() {
  # 앱을 백그라운드로 실행하고 로그로 리다이렉션
  # 사용법: start_streamlit <app_path> <port> [addr]
  local app_path="$1"
  local port="$2"
  local addr="${3:-0.0.0.0}"

  local log_dir="./runs/_streamlit_logs"
  mkdir -p "$log_dir"
  local name
  name="$(basename "$app_path" .py)"
  local out="${log_dir}/${name}.${port}.out.log"
  local err="${log_dir}/${name}.${port}.err.log"

  # 절대경로로 변환(작업 디렉터리 이동 대비)
  local app_abs
  app_abs="$(cd "$(dirname "$app_path")" && pwd)/$(basename "$app_path")"

  # 실행
  nohup "$PY" -m streamlit run "$app_abs" \
        --server.port "$port" \
        --server.address "$addr" \
        --server.headless "true" \
        --browser.gatherUsageStats "false" \
        >"$out" 2>"$err" &

  local pid=$!
  echo ">> [Streamlit] ${name} (port: ${port}) PID=${pid} (logs: ${out} / ${err})"
}

wait_listen() {
  # 지정 포트가 LISTEN 상태가 될 때까지 대기(기본 20초)
  # 사용법: wait_listen <port> [timeout_sec]
  local port="$1"
  local timeout="${2:-20}"

  for ((i=0; i<timeout; i++)); do
    if command -v ss >/dev/null 2>&1; then
      if ss -ltn "( sport = :$port )" | awk 'NR>1{print}' | grep -q .; then
        return 0
      fi
    elif command -v netstat >/dev/null 2>&1; then
      if netstat -tln 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$port$"; then
        return 0
      fi
    elif command -v lsof >/dev/null 2>&1; then
      if lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | awk '{print $9}' | grep -Eq "[:.]$port(->|$)"; then
        return 0
      fi
    else
      # 마지막 폴백: 포트 접속 시도(열리면 성공으로 간주)
      if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
        exec 3>&-
        return 0
      fi
    fi
    sleep 1
  done
  return 1
}

# ---------------------------------------------------------- #
#         섹션 2~7: 각종 자산 백테스트 및 검증 실행          #
# ---------------------------------------------------------- #

RUN_DIR_AAPL='./backtests/large_up_AAPL'
invoke_backtest --symbol 'AAPL' --out_dir "$RUN_DIR_AAPL" --source 'yahoo' --start "$START_DATE" --end "$END_DATE" --snapshot --calendar_id 'XNAS' --price_step '0.01'
invoke_validate "$RUN_DIR_AAPL"

RUN_DIR_INTC='./backtests/large_down_INTC'
invoke_backtest --symbol 'INTC' --out_dir "$RUN_DIR_INTC" --source 'yahoo' --start "$START_DATE" --end "$END_DATE" --snapshot --calendar_id 'XNAS' --price_step '0.01'
invoke_validate "$RUN_DIR_INTC"

RUN_DIR_KO='./backtests/large_sideways_KO'
invoke_backtest --symbol 'KO' --out_dir "$RUN_DIR_KO" --source 'yahoo' --start "$START_DATE" --end "$END_DATE" --snapshot --calendar_id 'XNYS' --price_step '0.01'
invoke_validate "$RUN_DIR_KO"

RUN_DIR_ENPH='./backtests/small_up_ENPH'
invoke_backtest --symbol 'ENPH' --out_dir "$RUN_DIR_ENPH" --source 'yahoo' --start "$START_DATE" --end "$END_DATE" --snapshot --calendar_id 'XNAS' --price_step '0.01'
invoke_validate "$RUN_DIR_ENPH"

RUN_DIR_GPRO='./backtests/small_down_GPRO'
invoke_backtest --symbol 'GPRO' --out_dir "$RUN_DIR_GPRO" --source 'yahoo' --start "$START_DATE" --end "$END_DATE" --snapshot --calendar_id 'XNAS' --price_step '0.01'
invoke_validate "$RUN_DIR_GPRO"

RUN_DIR_SIRI='./backtests/small_sideways_SIRI'
invoke_backtest --symbol 'SIRI' --out_dir "$RUN_DIR_SIRI" --source 'yahoo' --start "$START_DATE" --end "$END_DATE" --snapshot --calendar_id 'XNAS' --price_step '0.01'
invoke_validate "$RUN_DIR_SIRI"

# ---------------------------------------------------------- #
#                  섹션 8: 실시간 자동매매 실행               #
# ---------------------------------------------------------- #

echo ""
read -r -p ">> 모든 백테스트가 완료되었습니다. 실시간 자동매매를 시작하시겠습니까? (y/n) " confirmation

# RUNS_ROOT 부트스트랩: runs 디렉터리 보장 후 절대경로로 설정
mkdir -p ./runs
export RUNS_ROOT="$(cd ./runs && pwd)"
export SYNC_PROBE_URL="${SYNC_PROBE_URL:-https://www.google.com}"

liveProc_pid=""
if [[ "$confirmation" == "y" || "$confirmation" == "Y" ]]; then
  echo ">> [실시간 자동매매] 시작..."
  if [[ ! -f ".env" ]]; then
    echo "!! .env 파일이 없습니다. KIS/Upbit API 키와 계좌 정보를 설정해주세요." >&2
    exit 1
  fi

  TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
  LIVE_RUN_DIR="$RUNS_ROOT/live_run_dynamic_$TIMESTAMP"
  mkdir -p "$LIVE_RUN_DIR"
  echo ">> [실시간 자동매매] 결과 저장 경로: $LIVE_RUN_DIR"

  nohup "$PY" ./run_live.py \
      --config ./config.live.json \
      --out "$LIVE_RUN_DIR" \
      --source "auto" \
      --interval "1d" \
      --market "NASD" \
      --plan-out "$LIVE_RUN_DIR/plan" \
      --collect-fills \
      --fills-out "$LIVE_RUN_DIR/fills.jsonl" \
      --loop \
      > "$LIVE_RUN_DIR/live_bot.out.log" \
      2> "$LIVE_RUN_DIR/live_bot.err.log" &
  liveProc_pid=$!
  echo ">> [실시간 자동매매] PID = $liveProc_pid"
else
  echo ">> 실매매는 시작하지 않습니다. 모니터링만 기동합니다."
fi

# ---------------------------------------------------------- #
#            섹션 9~12: 모니터링 대시보드 동시 기동           #
# ---------------------------------------------------------- #

echo -e "\n>> [모니터링] 대시보드 실행..."
PORT_TRADE=8501
PORT_PORTFOL=8502
PORT_EQUITY=8503
PORT_SYSTEM=8504

start_streamlit ./monitoring/trade_log_viewer.py    "$PORT_TRADE"
start_streamlit ./monitoring/portfolio_viewer.py    "$PORT_PORTFOL"
start_streamlit ./monitoring/equity_curve_viewer.py "$PORT_EQUITY"
start_streamlit ./monitoring/system_tracker.py      "$PORT_SYSTEM"

echo ">> 접속 URL (EC2의 Public IP 주소로 접속하세요):"
echo "   - Trade Log Viewer     : http://<EC2_IP>:$PORT_TRADE"
echo "   - Portfolio Viewer     : http://<EC2_IP>:$PORT_PORTFOL"
echo "   - Equity Curve Viewer  : http://<EC2_IP>:$PORT_EQUITY"
echo "   - System Tracker       : http://<EC2_IP>:$PORT_SYSTEM"
if [[ -n "$liveProc_pid" ]]; then
  echo ">> 실매매 봇 PID         : $liveProc_pid"
fi

# 초기 레디니스 확인(기동 지연으로 인한 오탐 방지)
for p in "$PORT_TRADE" "$PORT_PORTFOL" "$PORT_EQUITY" "$PORT_SYSTEM"; do
  if wait_listen "$p" 20; then
    echo "[READY] 대시보드(Port: $p) 리슨 중입니다."
  else
    echo "[WARN] 대시보드(Port: $p)가 제한시간 내 열리지 않았습니다. 로그 확인: ./runs/_streamlit_logs" >&2
  fi
done

echo ">> 모든 프로세스가 백그라운드에서 실행되었습니다."

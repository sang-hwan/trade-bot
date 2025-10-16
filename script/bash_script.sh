# 1. 가상환경 활성화
# source .venv/bin/activate

# 2. 스크립트에 실행 권한 부여 (최초 1회만)
# chmod +x ./script/bash_script.sh

# 3. 스크립트 실행
# ./script/bash_script.sh

# ========================================================== #
#           자동매매 시스템 통합 제어 스크립트 (EC2용)       #
# ========================================================== #
set -e

# ---------------------------------------------------------- #
#                  공통 설정 및 유틸리티 함수                 #
# ---------------------------------------------------------- #
if [ -f ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

START_DATE='2018-01-01'
END_DATE='2025-09-30'

invoke_backtest() {
  echo ">> 백테스트 실행: $@"
  $PY ./backtest.py "$@"
  if [ $? -ne 0 ]; then echo "!! 백테스트 실패: $@" >&2; exit 1; fi
}

invoke_validate() {
  echo ">> 검증 실행: $1"
  $PY ./validate.py "$1"
  if [ $? -ne 0 ]; then echo "!! 검증 실패: $1" >&2; exit 1; fi
}

start_streamlit() {
  nohup $PY -m streamlit run "$1" --server.port "$2" --server.address "0.0.0.0" &
}

# ---------------------------------------------------------- #
#         섹션 2~7: 각종 자산 백테스트 및 검증 실행          #
# ---------------------------------------------------------- #
RUN_DIR_AAPL='./runs/large_up_AAPL'
invoke_backtest --symbol 'AAPL' --out_dir "$RUN_DIR_AAPL" --source 'yahoo' --start "$START_DATE" --end "$END_DATE" --snapshot --calendar_id 'XNAS' --price_step '0.01'
invoke_validate "$RUN_DIR_AAPL"

RUN_DIR_INTC='./runs/large_down_INTC'
invoke_backtest --symbol 'INTC' --out_dir "$RUN_DIR_INTC" --source 'yahoo' --start "$START_DATE" --end "$END_DATE" --snapshot --calendar_id 'XNAS' --price_step '0.01'
invoke_validate "$RUN_DIR_INTC"

RUN_DIR_KO='./runs/large_sideways_KO'
invoke_backtest --symbol 'KO' --out_dir "$RUN_DIR_KO" --source 'yahoo' --start "$START_DATE" --end "$END_DATE" --snapshot --calendar_id 'XNYS' --price_step '0.01'
invoke_validate "$RUN_DIR_KO"

RUN_DIR_ENPH='./runs/small_up_ENPH'
invoke_backtest --symbol 'ENPH' --out_dir "$RUN_DIR_ENPH" --source 'yahoo' --start "$START_DATE" --end "$END_DATE" --snapshot --calendar_id 'XNAS' --price_step '0.01'
invoke_validate "$RUN_DIR_ENPH"

RUN_DIR_GPRO='./runs/small_down_GPRO'
invoke_backtest --symbol 'GPRO' --out_dir "$RUN_DIR_GPRO" --source 'yahoo' --start "$START_DATE" --end "$END_DATE" --snapshot --calendar_id 'XNAS' --price_step '0.01'
invoke_validate "$RUN_DIR_GPRO"

RUN_DIR_SIRI='./runs/small_sideways_SIRI'
invoke_backtest --symbol 'SIRI' --out_dir "$RUN_DIR_SIRI" --source 'yahoo' --start "$START_DATE" --end "$END_DATE" --snapshot --calendar_id 'XNAS' --price_step '0.01'
invoke_validate "$RUN_DIR_SIRI"

# ---------------------------------------------------------- #
#                  섹션 8: 실시간 자동매매 실행               #
# ---------------------------------------------------------- #
echo ""
read -p ">> 모든 백테스트가 완료되었습니다. 실시간 자동매매를 시작하시겠습니까? (y/n) " confirmation

export RUNS_ROOT=$(realpath ./runs)
export SYNC_PROBE_URL="https://www.google.com"

liveProc_pid=""
if [[ "$confirmation" == "y" || "$confirmation" == "Y" ]]; then
  echo ">> [실시간 자동매매] 시작..."
  if [ ! -f ".env" ]; then echo "!! .env 파일이 없습니다." >&2; exit 1; fi

  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  LIVE_RUN_DIR="./runs/live_run_dynamic_$TIMESTAMP"
  mkdir -p "$LIVE_RUN_DIR"
  echo ">> [실시간 자동매매] 결과 저장 경로: $LIVE_RUN_DIR"

  nohup $PY ./run_live.py --config ./config.live.json --out "$LIVE_RUN_DIR" --source "auto" --interval "1d" --market "NASD" --plan-out "$LIVE_RUN_DIR/plan" --collect-fills --fills-out "$LIVE_RUN_DIR/fills.jsonl" --loop > "$LIVE_RUN_DIR/live_bot.log" 2>&1 &
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

start_streamlit ./monitoring/trade_log_viewer.py $PORT_TRADE
start_streamlit ./monitoring/portfolio_viewer.py $PORT_PORTFOL
start_streamlit ./monitoring/equity_curve_viewer.py $PORT_EQUITY
start_streamlit ./monitoring/system_tracker.py $PORT_SYSTEM

echo ">> 접속 URL (EC2의 Public IP 주소로 접속하세요):"
echo "   - Trade Log Viewer     : http://<EC2_IP>:$PORT_TRADE"
echo "   - Portfolio Viewer     : http://<EC2_IP>:$PORT_PORTFOL"
echo "   - Equity Curve Viewer  : http://<EC2_IP>:$PORT_EQUITY"
echo "   - System Tracker       : http://<EC2_IP>:$PORT_SYSTEM"

if [ -n "$liveProc_pid" ]; then
  echo ">> 실매매 봇 PID         : $liveProc_pid"
fi
echo ">> 모든 프로세스가 백그라운드에서 실행되었습니다."
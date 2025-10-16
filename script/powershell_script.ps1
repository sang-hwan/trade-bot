# ========================================================== #
#           자동매매 시스템 통합 제어 스크립트 (로컬 PC용)   #
# ========================================================== #
# 가상환경 활성화
# .\.venv\Scripts\Activate.ps1
# 사용 예:
#  1) 전체 실행(백테스트→검증→실매매·모니터링)
#     .\script\powershell_script.ps1
#  2) 실매매부터 바로 시작(백테스트/검증 스킵)
#     .\script\powershell_script.ps1 -LiveOnly

[CmdletBinding()]
param(
  [switch]$LiveOnly
)

# ---------------------------------------------------------- #
#                  공통 설정 및 유틸리티 함수                 #
# ---------------------------------------------------------- #
$ErrorActionPreference = 'Stop'

if (Test-Path ".\.venv\Scripts\python.exe") {
  $PY = ".\.venv\Scripts\python.exe"
} else {
  $PY = "python"
}

$START_DATE = '2018-01-01'
$END_DATE   = '2025-09-30'

function Invoke-Backtest {
  param([string[]]$ArgList)
  & $PY .\backtest.py @ArgList
  if ($LASTEXITCODE -ne 0) { throw "백테스트 실패" }
}

function Invoke-Validate {
  param([string]$RunDir)
  & $PY .\validate.py $RunDir
  if ($LASTEXITCODE -ne 0) { throw "검증 실패: $RunDir" }
}

function Start-Streamlit {
  param([string]$AppPath, [int]$Port, [string]$Addr = "0.0.0.0")
  Start-Process -FilePath $PY -ArgumentList @("-m","streamlit","run",$AppPath,"--server.port",$Port,"--server.address",$Addr) -WindowStyle Hidden -PassThru
}

# ---------------------------------------------------------- #
#         섹션 2~7: 각종 자산 백테스트 및 검증 실행          #
# ---------------------------------------------------------- #
if (-not $LiveOnly) {
  Write-Host ">> [AAPL 백테스트 및 검증] 시작..."
  $RUN_DIR_AAPL = '.\backtests\large_up_AAPL'
  Invoke-Backtest -ArgList @('--symbol','AAPL','--out_dir',$RUN_DIR_AAPL,'--source','yahoo','--start',$START_DATE,'--end',$END_DATE,'--snapshot','--calendar_id','XNAS','--price_step','0.01')
  Invoke-Validate -RunDir $RUN_DIR_AAPL
  Write-Host ">> [AAPL 백테스트 및 검증] 완료."

  Write-Host "`n>> [INTC 백테스트 및 검증] 시작..."
  $RUN_DIR_INTC = '.\backtests\large_down_INTC'
  Invoke-Backtest -ArgList @('--symbol','INTC','--out_dir',$RUN_DIR_INTC,'--source','yahoo','--start',$START_DATE,'--end',$END_DATE,'--snapshot','--calendar_id','XNAS','--price_step','0.01')
  Invoke-Validate -RunDir $RUN_DIR_INTC
  Write-Host ">> [INTC 백테스트 및 검증] 완료."

  Write-Host "`n>> [KO 백테스트 및 검증] 시작..."
  $RUN_DIR_KO = '.\backtests\large_sideways_KO'
  Invoke-Backtest -ArgList @('--symbol','KO','--out_dir',$RUN_DIR_KO,'--source','yahoo','--start',$START_DATE,'--end',$END_DATE,'--snapshot','--calendar_id','XNYS','--price_step','0.01')
  Invoke-Validate -RunDir $RUN_DIR_KO
  Write-Host ">> [KO 백테스트 및 검증] 완료."

  Write-Host "`n>> [ENPH 백테스트 및 검증] 시작..."
  $RUN_DIR_ENPH = '.\backtests\small_up_ENPH'
  Invoke-Backtest -ArgList @('--symbol','ENPH','--out_dir',$RUN_DIR_ENPH,'--source','yahoo','--start',$START_DATE,'--end',$END_DATE,'--snapshot','--calendar_id','XNAS','--price_step','0.01')
  Invoke-Validate -RunDir $RUN_DIR_ENPH
  Write-Host ">> [ENPH 백테스트 및 검증] 완료."

  Write-Host "`n>> [GPRO 백테스트 및 검증] 시작..."
  $RUN_DIR_GPRO = '.\backtests\small_down_GPRO'
  Invoke-Backtest -ArgList @('--symbol','GPRO','--out_dir',$RUN_DIR_GPRO,'--source','yahoo','--start',$START_DATE,'--end',$END_DATE,'--snapshot','--calendar_id','XNAS','--price_step','0.01')
  Invoke-Validate -RunDir $RUN_DIR_GPRO
  Write-Host ">> [GPRO 백테스트 및 검증] 완료."

  Write-Host "`n>> [SIRI 백테스트 및 검증] 시작..."
  $RUN_DIR_SIRI = '.\backtests\small_sideways_SIRI'
  Invoke-Backtest -ArgList @('--symbol','SIRI','--out_dir',$RUN_DIR_SIRI,'--source','yahoo','--start',$START_DATE,'--end',$END_DATE,'--snapshot','--calendar_id','XNAS','--price_step','0.01')
  Invoke-Validate -RunDir $RUN_DIR_SIRI
  Write-Host ">> [SIRI 백테스트 및 검증] 완료."
} else {
  Write-Host ">> [옵션] LiveOnly 모드: 백테스트/검증을 건너뜁니다."
}

# ---------------------------------------------------------- #
#                  섹션 8: 실시간 자동매매 실행               #
# ---------------------------------------------------------- #
Write-Host "`n"
if ($LiveOnly) {
  $confirmation = 'y'
} else {
  $confirmation = Read-Host ">> 모든 백테스트가 완료되었습니다. 실시간 자동매매를 시작하시겠습니까? (y/n)"
}

$env:RUNS_ROOT = (Resolve-Path ".\runs").Path
if (-not $env:SYNC_PROBE_URL) { $env:SYNC_PROBE_URL = "https://www.google.com" }

$liveProc = $null
if ($confirmation -eq 'y') {
  Write-Host ">> [실시간 자동매매] 시작..."
  if (-not (Test-Path -Path ".\.env")) { throw ".env 파일이 없습니다. KIS/Upbit API 키와 계좌 정보를 설정해주세요." }

  $TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
  $LIVE_RUN_DIR = ".\runs\live_run_dynamic_$TIMESTAMP"
  New-Item -ItemType Directory -Force -Path $LIVE_RUN_DIR | Out-Null
  Write-Host ">> [실시간 자동매매] 결과 저장 경로: $LIVE_RUN_DIR"

  $liveArgs = @(
    ".\run_live.py", "--config", ".\config.live.json", "--out", $LIVE_RUN_DIR,
    "--source", "auto", "--interval", "1d", "--market", "NASD",
    "--plan-out", (Join-Path $LIVE_RUN_DIR "plan"), "--collect-fills",
    "--fills-out", (Join-Path $LIVE_RUN_DIR "fills.jsonl"), "--loop"
  )

  $liveProc = Start-Process -FilePath $PY -ArgumentList $liveArgs -PassThru
  if (-not $liveProc) { throw "실매매 봇 실행 시작 실패" }
  Write-Host ">> [실매매] PID = $($liveProc.Id)"
} else {
  Write-Host ">> 실매매는 시작하지 않습니다. 모니터링만 기동합니다."
}

# ---------------------------------------------------------- #
#            섹션 9~12: 모니터링 대시보드 동시 기동           #
# ---------------------------------------------------------- #
Write-Host "`n>> [모니터링] 대시보드 실행..."
$PORT_TRADE   = 8501
$PORT_PORTFOL = 8502
$PORT_EQUITY  = 8503
$PORT_SYSTEM  = 8504

Start-Streamlit -AppPath ".\monitoring\trade_log_viewer.py"    -Port $PORT_TRADE
Start-Streamlit -AppPath ".\monitoring\portfolio_viewer.py"    -Port $PORT_PORTFOL
Start-Streamlit -AppPath ".\monitoring\equity_curve_viewer.py" -Port $PORT_EQUITY
Start-Streamlit -AppPath ".\monitoring\system_tracker.py"      -Port $PORT_SYSTEM

Write-Host ">> 접속 URL:"
Write-Host "   - Trade Log Viewer     : http://localhost:$PORT_TRADE"
Write-Host "   - Portfolio Viewer     : http://localhost:$PORT_PORTFOL"
Write-Host "   - Equity Curve Viewer  : http://localhost:$PORT_EQUITY"
Write-Host "   - System Tracker       : http://localhost:$PORT_SYSTEM"

if ($liveProc) {
  Write-Host ">> 실매매 봇 PID         : $($liveProc.Id)"
}

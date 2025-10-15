```powershell
# ========================================================== #
#           자동매매 시스템 통합 제어 스크립트               #
#                 (KIS + Upbit 동시 집행)                    #
# ========================================================== #
# 각 섹션은 독립적으로 실행될 수 있도록 구성되었습니다.
# 전체 스크립트를 실행하면 모든 백테스트 후 실매매 여부를 묻습니다.
# run_live.py는 유니버스 기반으로 후보를 스캔하고,
# 생성된 주문을 심볼 접두(KRX:/NASD:/NYSE:/KRW- 등)로 그룹핑하여
# KIS/Upbit 어댑터로 각각 집행합니다.
# ========================================================== #

# ---------------------------------------------------------- #
#                    섹션 1: 가상환경 관리                   #
# ---------------------------------------------------------- #
# 아래 명령어들은 최초 1회 또는 필요시에만 터미널에서 직접 실행합니다.

# 1-1. 가상환경 생성 (최초 1회):
python -m venv .venv

# 1-2. 가상환경 활성화 (스크립트 실행 전 항상 필요):
.\.venv\Scripts\Activate.ps1

# 1-3. 필요 라이브러리 설치 (최초 또는 라이브러리 추가 시):
pip install -r requirements.txt

# 1-4. 가상환경 종료 (모든 작업 완료 후):
deactivate


# ---------------------------------------------------------- #
#                  공통 설정 및 유틸리티 함수                 #
# ---------------------------------------------------------- #
$START_DATE = '2018-01-01'
$END_DATE   = '2025-09-30'

function Invoke-Backtest { 
  param([string[]]$ArgList)
  & python .\backtest.py @ArgList
  if ($LASTEXITCODE -ne 0) { throw "백테스트 실패" }
}

function Invoke-Validate { 
  param([string]$RunDir)
  & python .\validate.py $RunDir
  if ($LASTEXITCODE -ne 0) { throw "검증 실패: $RunDir" }
}


# ---------------------------------------------------------- #
#         섹션 2: 대형주 - 우상향 (AAPL) 백테스트 및 검증     #
# ---------------------------------------------------------- #
Write-Host ">> [AAPL 백테스트 및 검증] 시작..."
$RUN_DIR_AAPL = '.\runs\large_up_AAPL'
Invoke-Backtest -ArgList @('--symbol', 'AAPL', '--out_dir', $RUN_DIR_AAPL, '--source', 'yahoo', '--start', $START_DATE, '--end', $END_DATE, '--snapshot', '--calendar_id', 'XNAS')
Invoke-Validate -RunDir $RUN_DIR_AAPL
Write-Host ">> [AAPL 백테스트 및 검증] 완료. 결과 확인: $RUN_DIR_AAPL"


# ---------------------------------------------------------- #
#         섹션 3: 대형주 - 우하향 (INTC) 백테스트 및 검증     #
# ---------------------------------------------------------- #
Write-Host "`n>> [INTC 백테스트 및 검증] 시작..."
$RUN_DIR_INTC = '.\runs\large_down_INTC'
Invoke-Backtest -ArgList @('--symbol', 'INTC', '--out_dir', $RUN_DIR_INTC, '--source', 'yahoo', '--start', $START_DATE, '--end', $END_DATE, '--snapshot', '--calendar_id', 'XNAS')
Invoke-Validate -RunDir $RUN_DIR_INTC
Write-Host ">> [INTC 백테스트 및 검증] 완료. 결과 확인: $RUN_DIR_INTC"


# ---------------------------------------------------------- #
#          섹션 4: 대형주 - 박스권 (KO) 백테스트 및 검증       #
# ---------------------------------------------------------- #
Write-Host "`n>> [KO 백테스트 및 검증] 시작..."
$RUN_DIR_KO = '.\runs\large_sideways_KO'
Invoke-Backtest -ArgList @('--symbol', 'KO', '--out_dir', $RUN_DIR_KO, '--source', 'yahoo', '--start', $START_DATE, '--end', $END_DATE, '--snapshot', '--calendar_id', 'XNYS')
Invoke-Validate -RunDir $RUN_DIR_KO
Write-Host ">> [KO 백테스트 및 검증] 완료. 결과 확인: $RUN_DIR_KO"


# ---------------------------------------------------------- #
#         섹션 5: 소형주 - 우상향 (ENPH) 백테스트 및 검증     #
# ---------------------------------------------------------- #
Write-Host "`n>> [ENPH 백테스트 및 검증] 시작..."
$RUN_DIR_ENPH = '.\runs\small_up_ENPH'
Invoke-Backtest -ArgList @('--symbol', 'ENPH', '--out_dir', $RUN_DIR_ENPH, '--source', 'yahoo', '--start', $START_DATE, '--end', $END_DATE, '--snapshot', '--calendar_id', 'XNAS')
Invoke-Validate -RunDir $RUN_DIR_ENPH
Write-Host ">> [ENPH 백테스트 및 검증] 완료. 결과 확인: $RUN_DIR_ENPH"


# ---------------------------------------------------------- #
#         섹션 6: 소형주 - 우하향 (GPRO) 백테스트 및 검증     #
# ---------------------------------------------------------- #
Write-Host "`n>> [GPRO 백테스트 및 검증] 시작..."
$RUN_DIR_GPRO = '.\runs\small_down_GPRO'
Invoke-Backtest -ArgList @('--symbol', 'GPRO', '--out_dir', $RUN_DIR_GPRO, '--source', 'yahoo', '--start', $START_DATE, '--end', $END_DATE, '--snapshot', '--calendar_id', 'XNAS')
Invoke-Validate -RunDir $RUN_DIR_GPRO
Write-Host ">> [GPRO 백테스트 및 검증] 완료. 결과 확인: $RUN_DIR_GPRO"


# ---------------------------------------------------------- #
#         섹션 7: 소형주 - 박스권 (SIRI) 백테스트 및 검증     #
# ---------------------------------------------------------- #
Write-Host "`n>> [SIRI 백테스트 및 검증] 시작..."
$RUN_DIR_SIRI = '.\runs\small_sideways_SIRI'
Invoke-Backtest -ArgList @('--symbol', 'SIRI', '--out_dir', $RUN_DIR_SIRI, '--source', 'yahoo', '--start', $START_DATE, '--end', $END_DATE, '--snapshot', '--calendar_id', 'XNAS')
Invoke-Validate -RunDir $RUN_DIR_SIRI
Write-Host ">> [SIRI 백테스트 및 검증] 완료. 결과 확인: $RUN_DIR_SIRI"


# ---------------------------------------------------------- #
#                  섹션 8: 실시간 자동매매 실행               #
# ---------------------------------------------------------- #
Write-Host "`n"
$confirmation = Read-Host ">> 모든 백테스트가 완료되었습니다. 실시간 자동매매를 시작하시겠습니까? (y/n)"

if ($confirmation -eq 'y') {
  Write-Host ">> [실시간 자동매매] 시작..."

  if (-not (Test-Path -Path ".\.env")) {
    throw ".env 파일이 없습니다. KIS/Upbit API 키와 계좌 정보를 설정해주세요."
  }

  # 결과 디렉터리
  $TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
  $LIVE_RUN_DIR = ".\runs\live_run_dynamic_$TIMESTAMP"
  Write-Host ">> [실시간 자동매매] 결과 저장 경로: $LIVE_RUN_DIR"

  # 중요: run_live.py는 유니버스 기반 스캐닝과 멀티 브로커 라우팅을 수행합니다.
  # --source는 'auto'로 지정(자산군별 내부 라우팅), --interval은 1d(코인은 09:00 KST 앵커 적용)
  & python .\run_live.py @(
    '--config',   '.\config.live.json',
    '--out',      $LIVE_RUN_DIR,
    '--source',   'auto',
    '--interval', '1d',
    '--market',   'NASD',                # KIS 체결 수집 시 사용(Upbit는 헤더 기반 레이트리밋 감사만 기록)
    '--plan-out', "$LIVE_RUN_DIR\plan",
    '--collect-fills',
    '--fills-out', "$LIVE_RUN_DIR\fills.jsonl",
    '--loop'
  )

  if ($LASTEXITCODE -ne 0) { throw "실매매 봇 실행 중 오류 발생" }
  Write-Host ">> [실시간 자동매매] 정상 종료."

} else {
  Write-Host ">> 실매매를 시작하지 않고 스크립트를 종료합니다."
}

# ---------------------------------------------------------- #
#            섹션 9: 모니터링 - 매매 로그 시각화 뷰어         #
# ---------------------------------------------------------- #
Write-Host "`n>> [모니터링] Trade Log Viewer 실행..."

# runs 루트 경로 지정 (필요시 변경)
$env:RUNS_ROOT = (Resolve-Path ".\runs").Path

# 가상환경이 활성화되어 있어야 함 (상단 섹션 1-2 참고)
# 포트는 필요시 변경 가능 (기본 8501)
python -m streamlit run .\monitoring\trade_log_viewer.py --server.port 8501 --server.address 0.0.0.0

if ($LASTEXITCODE -ne 0) { throw "Trade Log Viewer 실행 중 오류 발생" }
```
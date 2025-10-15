```powershell
# ---------- # 섹션: 가상환경
# 스크립트 실행 전, 가상환경 활성화 및 requirements.txt 설치가 필요합니다.
.\.venv\Scripts\Activate.ps1

# ---------- # 섹션: 공통 변수
$START_DATE = '2018-01-01' # 백테스트 시작일 (UTC)
$END_DATE   = '2025-09-30' # 백테스트 종료일 (UTC)

# ---------- # 섹션: 유틸리티 함수
# backtest.py 실행 및 오류 시 중단
function Invoke-Backtest {
    param([string[]]$ArgList)
    & python .\backtest.py @ArgList; if ($LASTEXITCODE -ne 0) { throw "백테스트 실패 (종료 코드: $LASTEXITCODE)" }
}
# validate.py 실행 및 오류 시 중단
function Invoke-Validate {
    param([Parameter(Mandatory=$true)][string]$RunDir)
    & python .\validate.py $RunDir; if ($LASTEXITCODE -ne 0) { throw "검증 실패: $RunDir (종료 코드: $LASTEXITCODE)" }
}

# ---------- # 섹션: AAPL 백테스트 (대형주)
Write-Host ">> (1/3) AAPL 백테스트 및 검증 시작..."
$AAPL_RUN_DIR = '.\runs\AAPL_backtest_run' # 결과물 저장 디렉터리

$ArgsAAPL = @(
  # --- 데이터 및 기간 ---
  '--source',        'yahoo',         # 데이터 소스 ('yahoo' or 'upbit')
  '--symbol',        'AAPL',          # 백테스트 대상 심볼
  '--start',         $START_DATE,     # 시뮬레이션 시작일 (UTC)
  '--end',           $END_DATE,       # 시뮬레이션 종료일 (UTC)
  '--interval',      '1d',            # 데이터 시간 주기 (일봉)
  # --- 전략 핵심 파라미터 ---
  '--N',             '20',            # 돈치안 채널 기간 (스탑/사이징용)
  '--f',             '0.02',          # 거래당 리스크 비율 (자본의 2%)
  '--epsilon',       '0.0',           # SMA 교차 신호의 최소 이격 조건
  # --- 거래 환경 및 비용 ---
  '--initial_equity',  '1000000',     # 초기 자본금
  '--commission_rate', '0.001',       # 거래 수수료 (0.1%)
  '--slip',            '0.0005',      # 슬리피지 (0.05%)
  # --- 자산 및 주문 단위 ---
  '--lot_step',        '1',           # 최소 주문 수량 (1주)
  '--price_step',      '0.01',        # 최소 가격 단위 ($0.01)
  '--base_currency',   'USD',         # 기준 통화
  '--calendar_id',     'XNAS',        # 거래일 캘린더 (나스닥)
  # --- 출력 및 기타 ---
  '--snapshot',                      # 데이터 스냅샷 생성 플래그
  '--out_dir',         $AAPL_RUN_DIR # 결과물 저장 경로
)
Invoke-Backtest -ArgList $ArgsAAPL
Invoke-Validate -RunDir $AAPL_RUN_DIR
Write-Host ">> (1/3) AAPL 완료. 결과 확인: $AAPL_RUN_DIR"

# ---------- # 섹션: SIRI 백테스트 (박스권 소형주)
Write-Host ">> (2/3) SIRI 백테스트 및 검증 시작..."
$SIRI_RUN_DIR = '.\runs\SIRI_backtest_run' # 결과물 저장 디렉터리

$ArgsSIRI = $ArgsAAPL.Clone() # AAPL 인자 복사 후 일부만 수정
$ArgsSIRI[3] = 'SIRI'          # --symbol 인자 변경
$ArgsSIRI[32] = $SIRI_RUN_DIR  # --out_dir 인자 변경

Invoke-Backtest -ArgList $ArgsSIRI
Invoke-Validate -RunDir $SIRI_RUN_DIR
Write-Host ">> (2/3) SIRI 완료. 결과 확인: $SIRI_RUN_DIR"

# ---------- # 섹션: GPRO 백테스트 (하향 소형주)
Write-Host ">> (3/3) GPRO 백테스트 및 검증 시작..."
$GPRO_RUN_DIR = '.\runs\GPRO_backtest_run' # 결과물 저장 디렉터리

$ArgsGPRO = $ArgsAAPL.Clone() # AAPL 인자 복사 후 일부만 수정
$ArgsGPRO[3] = 'GPRO'          # --symbol 인자 변경
$ArgsGPRO[32] = $GPRO_RUN_DIR  # --out_dir 인자 변경

Invoke-Backtest -ArgList $ArgsGPRO
Invoke-Validate -RunDir $GPRO_RUN_DIR
Write-Host ">> (3/3) GPRO 완료. 결과 확인: $GPRO_RUN_DIR"

# ---------- # 섹션: 종료
deactivate
```
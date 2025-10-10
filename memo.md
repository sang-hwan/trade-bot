```powershell
# --------------------------- # 섹션: 가상환경 & 의존성
python -m venv .venv                           # 가상환경 생성
.\.venv\Scripts\Activate.ps1                   # 가상환경 활성화(Windows PowerShell)

python -m pip install --upgrade pip            # pip 최신화
pip install -r requirements.txt                # 프로젝트 의존성 설치

# --------------------------- # 섹션: 공통 기간(UTC)
$START = '2018-01-01'                          # 시작일(UTC, ISO-8601)
$END   = '2025-09-30'                          # 종료일(UTC, ISO-8601)

# --------------------------- # 섹션: 유틸 함수
function Invoke-Backtest { param([string[]]$ArgList) & python .\backtest.py @ArgList; if ($LASTEXITCODE -ne 0) { throw "Backtest failed (exit $LASTEXITCODE)" } }  # 백테스트 실행+실패시 중단
function Invoke-Validate { param([Parameter(Mandatory=$true)][string]$RunDir, [string]$DocsDir = $null) if ($null -ne $DocsDir -and $DocsDir -ne '') { & python .\validate.py $RunDir --docs-dir $DocsDir } else { & python .\validate.py $RunDir }; if ($LASTEXITCODE -ne 0) { throw "Validation failed: $RunDir (exit $LASTEXITCODE)" } }  # 검증 실행+실패시 중단

# --------------------------- # 섹션: backtest.py — AAPL (Yahoo)
$AAPL_OUT = '.\runs\AAPL_Donchian_N20_F02'     # 출력 디렉터리
$ArgsAAPL = @(                                 # AAPL 백테스트 인자 배열 시작
  '--source','yahoo',                          # 데이터 소스
  '--symbol','AAPL',                           # 심볼(티커)
  '--start',$START,                            # 시작일(UTC, YYYY-MM-DD)
  '--end',$END,                                # 종료일(UTC)
  '--interval','1d',                           # 봉 주기(일봉)
  '--N','20',                                  # Donchian N
  '--f','0.02',                                # 위험비율 f
  '--lot_step','1',                            # 수량 라운딩(주식=1주)
  '--price_step','0.01',                       # 가격 라운딩(미국 주식 최소틱)
  '--commission_rate','0.001',                 # 수수료율
  '--slip','0.0005',                           # 슬리피지율
  '--epsilon','0',                             # 타이브레이크 ε
  '--initial_equity','10000000',               # 초기자본
  '--base_currency','USD',                     # 기준 통화
  '--calendar_id','XNAS',                      # 거래 캘린더 ID
  '--snapshot',                                # 스냅샷 저장 플래그
  '--out_dir',$AAPL_OUT                        # 출력 디렉터리 지정
)                                              # AAPL 인자 배열 끝
Invoke-Backtest -ArgList $ArgsAAPL             # AAPL 백테스트 실행
Invoke-Validate -RunDir $AAPL_OUT              # AAPL 결과 검증 및 리포트(필요 시 -DocsDir .\docs)

# --------------------------- # 섹션: backtest.py — SIRI (Yahoo)
$SIRI_OUT = '.\runs\SIRI_Donchian_N20_F02'     # 출력 디렉터리
$ArgsSIRI = @(                                 # SIRI 백테스트 인자 배열 시작
  '--source','yahoo',                          # 데이터 소스
  '--symbol','SIRI',                           # 심볼(티커)
  '--start',$START,                            # 시작일(UTC)
  '--end',$END,                                # 종료일(UTC)
  '--interval','1d',                           # 봉 주기(일봉)
  '--N','20',                                  # Donchian N
  '--f','0.02',                                # 위험비율 f
  '--lot_step','1',                            # 수량 라운딩(주식=1주)
  '--price_step','0.01',                       # 가격 라운딩(센트)
  '--commission_rate','0.001',                 # 수수료율
  '--slip','0.0005',                           # 슬리피지율
  '--epsilon','0',                             # 타이브레이크 ε
  '--initial_equity','10000000',               # 초기자본
  '--base_currency','USD',                     # 기준 통화
  '--calendar_id','XNAS',                      # 거래 캘린더 ID
  '--snapshot',                                # 스냅샷 저장 플래그
  '--out_dir',$SIRI_OUT                        # 출력 디렉터리 지정
)                                              # SIRI 인자 배열 끝
Invoke-Backtest -ArgList $ArgsSIRI             # SIRI 백테스트 실행
Invoke-Validate -RunDir $SIRI_OUT              # SIRI 결과 검증 및 리포트

# --------------------------- # 섹션: backtest.py — GPRO (Yahoo)
$GPRO_OUT = '.\runs\GPRO_Donchian_N20_F02'     # 출력 디렉터리
$ArgsGPRO = @(                                 # GPRO 백테스트 인자 배열 시작
  '--source','yahoo',                          # 데이터 소스
  '--symbol','GPRO',                           # 심볼(티커)
  '--start',$START,                            # 시작일(UTC)
  '--end',$END,                                # 종료일(UTC)
  '--interval','1d',                           # 봉 주기(일봉)
  '--N','20',                                  # Donchian N
  '--f','0.02',                                # 위험비율 f
  '--lot_step','1',                            # 수량 라운딩(주식=1주)
  '--price_step','0.01',                       # 가격 라운딩(센트)
  '--commission_rate','0.001',                 # 수수료율
  '--slip','0.0005',                           # 슬리피지율
  '--epsilon','0',                             # 타이브레이크 ε
  '--initial_equity','10000000',               # 초기자본
  '--base_currency','USD',                     # 기준 통화
  '--calendar_id','XNAS',                      # 거래 캘린더 ID
  '--snapshot',                                # 스냅샷 저장 플래그
  '--out_dir',$GPRO_OUT                        # 출력 디렉터리 지정
)                                              # GPRO 인자 배열 끝
Invoke-Backtest -ArgList $ArgsGPRO             # GPRO 백테스트 실행
Invoke-Validate -RunDir $GPRO_OUT              # GPRO 결과 검증 및 리포트



# --------------------------- # 섹션: 가상환경 비활성화
deactivate                                      # 가상환경 비활성화
```

---

```powershell
# --------------------------- # 섹션: 가상환경
.\.venv\Scripts\Activate.ps1

# =========================== 안정화 래퍼 (미국장 세션 전용) ===========================
# - 콘솔 즉시 종료 방지, 오류 표시 강화, 작업 디렉터리 고정
# - 미국 정규장(ET 09:30–16:00) 동안만 run_live.py --loop 실행
# - 개장 5분 전 시작, 마감+5분까지 유지 (DST 자동 반영)

$ErrorActionPreference = 'Stop'

# 안전한 루트 탐지: PSScriptRoot → MyInvocation → 현재 폴더
if ($PSScriptRoot -and $PSScriptRoot.Trim()) {
  $ROOT = $PSScriptRoot
} elseif ($MyInvocation.MyCommand.Path) {
  $ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
  $ROOT = (Get-Location).Path
}
Set-Location $ROOT

function Invoke-Live {
  param([string[]]$ArgList)
  & python .\run_live.py @ArgList
  $exit = $LASTEXITCODE
  if ($exit -ne 0) { Write-Error "run_live.py failed (exit $exit)" }
  return $exit
}

try {
  # --- 경로/출력 준비 ---
  $DATE      = Get-Date -Format 'yyyyMMdd'
  $RUN_DIR   = ".\live_runs\${DATE}_NASD_TNDM_CWCO"
  $PLAN_DIR  = "$RUN_DIR\plan"
  $CFG_PATH  = ".\config.live.json"
  $FILLS_LOG = "$RUN_DIR\fills.jsonl"

  New-Item -ItemType Directory -Force -Path $RUN_DIR  | Out-Null
  New-Item -ItemType Directory -Force -Path $PLAN_DIR | Out-Null

  if (-not (Test-Path .\run_live.py)) { throw "run_live.py not found in $((Get-Location).Path)" }

  # --- 타임존 계산 (DST 자동), 주말 스킵 ---
  $tzET  = [System.TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
  $tzKST = [System.TimeZoneInfo]::FindSystemTimeZoneById('Korea Standard Time')

  $nowUtc = (Get-Date).ToUniversalTime()
  $nowET  = [System.TimeZoneInfo]::ConvertTimeFromUtc($nowUtc, $tzET)

  if ($nowET.DayOfWeek -in @('Saturday','Sunday')) {
    Write-Host "[info] U.S. market closed today (weekend). Exiting."
    exit 0
  }

  # ET “벽시각”을 Kind=Unspecified로 생성 후 ET→KST 변환
  $openET  = [datetime]::new($nowET.Year, $nowET.Month, $nowET.Day, 9, 30, 0)   # 09:30 (ET)
  $closeET = [datetime]::new($nowET.Year, $nowET.Month, $nowET.Day, 16, 0, 0)   # 16:00 (ET)

  $openKST  = [System.TimeZoneInfo]::ConvertTime($openET,  $tzET, $tzKST)
  $closeKST = [System.TimeZoneInfo]::ConvertTime($closeET, $tzET, $tzKST)

  # 개장 5분 전 시작, 마감+5분까지 유지
  $startAt = $openKST.AddMinutes(-5)
  $endAt   = $closeKST.AddMinutes(5)
  $nowKST  = [System.TimeZoneInfo]::ConvertTimeFromUtc($nowUtc, $tzKST)

  if ($nowKST -lt $startAt) {
    $waitSec = [int]([TimeSpan]($startAt - $nowKST)).TotalSeconds
    Write-Host "[info] Waiting until $($startAt.ToString('yyyy-MM-dd HH:mm:ss K')) KST to start..."
    Start-Sleep -Seconds $waitSec
    # 대기 후 현재시각 갱신
    $nowUtc = (Get-Date).ToUniversalTime()
  } elseif ($nowKST -ge $endAt) {
    Write-Host "[info] U.S. session already finished for today. Exiting."
    exit 0
  }

  # 최대 실행 시간(초): 지금~(마감+5분)
  $nowKST = [System.TimeZoneInfo]::ConvertTimeFromUtc($nowUtc, $tzKST)
  $maxRuntimeSec = [int]([TimeSpan]($endAt - $nowKST)).TotalSeconds
  if ($maxRuntimeSec -le 0) { $maxRuntimeSec = 60 }

  # --- run_live 실행 (루프 모드) ---
  $ArgsLive = @(
    '--config',   $CFG_PATH,
    '--out',      $RUN_DIR,
    '--source',   'yahoo',
    '--interval', '1d',
    '--plan-out', $PLAN_DIR,
    '--collect-fills',
    '--market',   'NASD',
    '--fills-out', $FILLS_LOG,
    '--loop',
    '--poll-fills-every', '5',     # 체결 폴링 주기(초)
    '--reconcile-every',  '30',    # 대사/산출물 주기(초)
    '--max-runtime',      "$maxRuntimeSec"
  )

  $code = Invoke-Live -ArgList $ArgsLive
  if ($code -ne 0) { exit $code }
}
catch {
  Write-Error $_
}
finally {
  if ($Host.Name -match 'ConsoleHost') {
    [void](Read-Host "Press ENTER to close")
  }
}

# --------------------------- # 섹션: 가상환경 비활성화
deactivate
```
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
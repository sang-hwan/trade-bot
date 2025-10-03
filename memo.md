```powershell
# 가상환경 생성
python -m venv .venv


# 가상환경 활성화
.\.venv\Scripts\Activate.ps1


# 필요 라이브러리 설치
python -m pip install --upgrade pip
pip install -r requirements.txt

# 공통 기간(UTC)
$START = '2018-01-01'
$END   = '2025-09-30'

function Invoke-Backtest { param([string[]]$ArgList) & python .\main.py @ArgList }

# main.py 실행 — 해외주식 (AAPL, Yahoo)
$AAPL_OUT = '.\runs\AAPL_Donchian_N20_F02'
$ArgsAAPL = @(
  '--source','yahoo',                           # 데이터 소스
  '--symbol','AAPL',                            # 심볼(티커)
  '--start',$START,                             # 시작일(UTC, YYYY-MM-DD)
  '--end',$END,                                 # 종료일(UTC)
  '--interval','1d',                            # 봉 주기(일봉)
  '--N','20',                                   # Donchian N
  '--f','0.02',                                 # 위험비율 f
  '--lot_step','1',                             # 수량 라운딩(주식=1주)
  '--price_step','0.01',                        # 가격 라운딩(미국 주식 최소틱)
  '--commission_rate','0.001',                  # 수수료율
  '--slip','0.0005',                            # 슬리피지율
  '--epsilon','0',                              # 타이브레이크 ε
  '--initial_equity','10000000',                # 초기자본
  '--base_currency','USD',                      # 기준 통화
  '--calendar_id','XNAS',                       # 거래 캘린더 ID
  '--snapshot',                                 # 스냅샷 저장
  '--out_dir',$AAPL_OUT                         # 출력 디렉터리
)
Invoke-Backtest -ArgList $ArgsAAPL

# main.py 실행 — 해외주식 (SIRI, Yahoo)
$SIRI_OUT = '.\runs\SIRI_Donchian_N20_F02'
$ArgsSIRI = @(
  '--source','yahoo',
  '--symbol','SIRI',
  '--start',$START,
  '--end',$END,
  '--interval','1d',
  '--N','20',
  '--f','0.02',
  '--lot_step','1',          # 주식= 1주 단위
  '--price_step','0.01',     # 기본 최소틱(센트). 필요시 0.005/0.0001로 민감도 실험
  '--commission_rate','0.001',
  '--slip','0.0005',
  '--epsilon','0',
  '--initial_equity','10000000',
  '--base_currency','USD',
  '--calendar_id','XNAS',
  '--snapshot',
  '--out_dir',$SIRI_OUT
)
Invoke-Backtest -ArgList $ArgsSIRI

# main.py 실행 — 해외주식 (GPRO, Yahoo)
$GPRO_OUT = '.\runs\GPRO_Donchian_N20_F02'
$ArgsGPRO = @(
  '--source','yahoo',
  '--symbol','GPRO',
  '--start',$START,
  '--end',$END,
  '--interval','1d',
  '--N','20',
  '--f','0.02',
  '--lot_step','1',          # 주식= 1주 단위
  '--price_step','0.01',     # 필요시 0.005/0.0001로 조정해 테스트
  '--commission_rate','0.001',
  '--slip','0.0005',
  '--epsilon','0',
  '--initial_equity','10000000',
  '--base_currency','USD',
  '--calendar_id','XNAS',
  '--snapshot',
  '--out_dir',$GPRO_OUT
)
Invoke-Backtest -ArgList $ArgsGPRO

# 비활성화
deactivate
```

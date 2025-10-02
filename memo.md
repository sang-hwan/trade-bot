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
  '--lot_step','0.01',                          # 라운딩 단위
  '--commission_rate','0.001',                  # 수수료율
  '--slip','0.0005',                            # 슬리피지율
  '--epsilon','0',                              # 타이브레이크 ε
  '--initial_equity','10000000',                # 초기자본
  '--snapshot',                                 # 스냅샷 저장
  '--out_dir',$AAPL_OUT                         # 출력 디렉터리
)
Invoke-Backtest -ArgList $ArgsAAPL

# main.py 실행 — 암호화폐 (BTC/KRW, Upbit)
$BTC_OUT = '.\runs\BTC_Donchian_N55_F015'
$ArgsBTC = @(
  '--source','upbit',                           # 데이터 소스
  '--symbol','BTC/KRW',                         # 심볼(실행 시 KRW-BTC로 정규화)
  '--start',$START,                             # 시작일(UTC)
  '--end',$END,                                 # 종료일(UTC)
  '--interval','1d',                            # 인터벌(실행 시 day로 정규화)
  '--N','55',                                   # Donchian N
  '--f','0.015',                                # 위험비율 f
  '--lot_step','1000',                          # 라운딩 단위(KRW)
  '--commission_rate','0.0005',                 # 수수료율
  '--slip','0.0007',                            # 슬리피지율
  '--epsilon','0',                              # 타이브레이크 ε
  '--initial_equity','10000000',                # 초기자본
  '--snapshot',                                 # 스냅샷 저장
  '--out_dir',$BTC_OUT                          # 출력 디렉터리
)
Invoke-Backtest -ArgList $ArgsBTC

# validation.py 실행 (validate_run.py)
$ValidateArgs = @(
  '--run_dir',$AAPL_OUT,                        # main.py 산출물 디렉터리
  '--out_dir','.\\validation_report',           # 리포트 출력 디렉터리
  '--train_years','3',                          # 학습 윈도(연)
  '--test_years','1',                           # 검증 윈도(연)
  '--step_years','1',                           # 롤링 스텝(연)
  '--top_k','20',                               # 로버스트 상위 N
  '--top_k_curves','5'                          # 자본곡선 PNG 수
)
python .\validate_run.py @ValidateArgs

# 비활성화
deactivate
```

---

1. **수식 오탈자 전면 교정**

   * 내용: `*` 제거, `\{ \}` 집합 표기, 줄바꿈 `\\[6pt]/\\[8pt]`, `\in \{\mathrm{True},\mathrm{False}\}`, `$D\le 0 \Rightarrow$` 문구 정리
   * 섹션: **1) 데이터 > 데이터 정합성**, **2) 전략 > 2.1 특징, 2.2 의사결정, 2.4 리밸런싱**, **3) 시뮬레이션 > 기준 통화 평가, 체결·비용·사이징 반영**

2. **라운딩 용어 통일 (`lot_step` / `price_step`)**

   * 내용: “호가 단위” 문구를 `price_step`로, 수량 라운딩은 `lot_step`로 고정
   * 섹션: **2) 전략 > 2.3 사이징**, **3) 시뮬레이션 > 체결·비용·사이징 반영**

3. **시간 표기 통일 ($t-1$ 형식)**

   * 내용: `t!-!1` 등을 `$t-1$`(또는 `$t\!-\!1$`)로 일관
   * 섹션: **2) 전략 > 2.4 리밸런싱**, **3) 시뮬레이션 > 역할/타임라인**

4. **텍스트 변수의 수식 내 표기 통일**

   * 내용: `\texttt{Open\_eff}`, `\texttt{Position}_{i,t}` 등 언더스코어 이스케이프 및 `\texttt{}` 사용
   * 섹션: **3) 시뮬레이션 > 체결·비용·사이징 반영**, **3) 시뮬레이션 > 기준 통화 평가**

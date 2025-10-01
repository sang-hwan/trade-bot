```powershell
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
.\.venv\Scripts\Activate.ps1

# 비활성화
deactivate

# 필요 라이브러리 설치
pip install -r requirements.txt

# 해외주식 (AAPL, Yahoo)
$ArgsAAPL = @(
  '--source','yahoo'                     # 데이터 소스: Yahoo Finance
  '--symbol','AAPL'                      # 종목 심볼
  '--start','2018-01-01'                 # 수집 시작일(UTC)
  '--end','2025-09-30'                   # 수집 종료일
  '--interval','1d'                      # 봉 주기: 일봉
  '--N','20','--f','0.02','--lot_step','0.01'      # Donchian N=20, 위험비율 f=2%, 라운딩 단위 0.01
  '--commission_rate','0.001','--slip','0.0005','--epsilon','0'  # 수수료 0.1%, 슬리피지 0.05%, ε=0
  '--initial_equity','10000000'          # 초기자본
  '--snapshot'                           # 스냅샷 저장 활성화(Parquet+SHA-256)
  '--out_dir','.\runs\AAPL_Donchian_N20_F02'       # 출력 디렉터리
)
python .\main.py @ArgsAAPL                 # 엔진 실행

# 암호화폐 (BTC/KRW, Upbit)
$ArgsBTC = @(
  '--source','upbit'                      # 데이터 소스: Upbit Public API
  '--symbol','BTC/KRW'                    # 심볼(실행 시 'KRW-BTC'로 자동 정규화)
  '--start','2018-01-01'                  # 수집 시작일(UTC)
  '--end','2025-09-30'                    # 수집 종료일
  '--interval','1d'                       # 인터벌(실행 시 'day'로 자동 정규화)
  '--N','55','--f','0.015','--lot_step','1000'     # Donchian N=55, f=1.5%, 라운딩 단위 1,000(KRW)
  '--commission_rate','0.0005','--slip','0.0007','--epsilon','0' # 수수료 0.05%, 슬리피지 0.07%, ε=0
  '--initial_equity','10000000'           # 초기자본
  '--snapshot'                            # 스냅샷 저장 활성화
  '--out_dir','.\runs\BTC_Donchian_N55_F015'       # 출력 디렉터리
)
python .\main.py @ArgsBTC                  # 엔진 실행

---

## 개발 프로젝트 파일 목록

### data/

* `collect.py` — 원천(CSV/API) 로드 → 인덱스 `DatetimeIndex[UTC]` 정규화 → 컬럼 매핑(`open, high, low, close`)
* `quality_gate.py` — 단조 증가/중복/음수 간격/바 무결성 검증, 위반 시 즉시 실패
* `adjust.py` — 보정계수 `AdjClose/Close`로 `open_adj, high_adj, low_adj, close_adj` 생성(있으면 **항상 이 컬럼만 사용**)
* `snapshot.py` — **Parquet** 저장, **SHA-256** 해시 계산, 스냅샷 메타(`source, symbol, start, end, interval, rows, columns, snapshot_path, snapshot_sha256, collected_at, timezone`) 기록

> ※ 비용·체결 관련 값(수수료율, 슬리피지, 호가/로트 단위 등)은 여기서는 **참조 상수 보관만** 하고, 계산은 하지 않습니다.

---

### strategy/

* `features.py` — `SMA_n`, Donchian **이전 N-저가 기준선** `prev_low_N(t-1)` 생성(입력은 `*_adj` 우선)
* `signals.py` — **SMA10/50 크로스(롱 온리)** 신호 결정: `signal_next` (결정=Close(t-1), 체결=Open(t), 타이브레이크 ε)
* `stops.py` — **Donchian N-저가 이탈(롱 기준)** 스탑 판정: `stop_hit`, `stop_level=prev_low_N(t)` (판정=Close(t), 체결=Open(t+1))
* `sizing_spec.py` — **Fixed Fractional 스펙 산출 전용**(f, `stop_level`, lot/호가 단위 등 **스펙만** 반환; 수량/체결 계산은 금지)

> ※ 전략 단계는 **정합화·스냅샷 저장 금지**, **실제 수량/비용 계산 금지**(룩어헤드 방지).

---

### simulation/

* `engine.py` — 타임라인 루프(**On-Open 집행 → On-Close 판정**), **스탑 > 신호** 우선순위 적용
* `execution.py` — 체결가 계산(`Open_eff= open_adj` 우선, 없으면 `open`; 매수 `×(1+slip)`, 매도 `×(1−slip)`), 수수료 반영
* `sizing_on_open.py` — **On-Open**에서만 위험예산/스톱거리로 수량 `Q` 계산 → 실행 수량 `Q_exec=⌊Q/s⌋·s`(하향 라운딩)
* `outputs.py` — 산출물 고정: `trades`, `equity_curve`, `metrics`, `run_meta`(파라미터·스냅샷 해시·버전 포함)

> ※ 시뮬레이션은 **데이터 정합/특징/의사결정 재구현 금지**, 오직 전략 호출과 집행·비용 반영만 담당.

---

### validation/

* `splits.py` — 표본내/외 분리, **워크포워드** 롤링 분할(예: 3y train → 1y test)
* `grid.py` — 민감도/스트레스 파라미터 그리드 생성(`n, N, f, commission, slip …`)
* `evaluate.py` — 조합별 시뮬레이션 실행 및 `metrics` 집계(IQR·최악 포함 로버스트 구간 식별)
* `report.py` — 리포팅/시각화(자본곡선, 히트맵, 베스트 vs 로버스트 하이라이트)

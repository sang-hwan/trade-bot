# 목표

데이터[수집 → 정합성 → 완전 조정 → 스냅샷 고정] → 전략[피처 → 신호·스탑 → 사이징] → 시뮬레이션[타임라인 집행 · 비용 반영 → 산출물 고정] → 검증·리포팅[워크포워드 · 민감도 · 시각화]

> 위 순서를 기준으로 **SSOT(역할 단일화)** 를 준수하고, **룩어헤드 금지**, **Stop > Signal** 우선, `*_adj` **우선 사용**을 기본 계약으로 하여 **실행·재현성·비용 반영**을 일관되게 유지한다.

---

## 아키텍처 원칙(SSOT)
- 데이터 정합화는 **데이터 단계**에만, 특징 생성·의사결정은 **전략 단계**에만, 타임라인·체결·비용·우선순위는 **시뮬레이션 단계**에만 구현한다.
- 동일 규칙의 중복 구현을 금지한다.

---

## 표기 원칙(일관성)
- 용어: **특징(feature)**, **의사결정(decision)**, **스냅샷 메타(snapshot meta)**.
- 조정가: `*_adj` 컬럼이 존재하면 **항상 우선 사용**(혼용 금지).
- 기호 매핑: $O \to \texttt{open}$, $H \to \texttt{high}$, $L \to \texttt{low}$, $C \to \texttt{close}$, $\mathrm{AdjClose} \to \texttt{AdjClose}$.
- 시계열 표기:
  - 시점 $t$ 값은 하첨자(예: $L_t$).
  - Donchian 기준선(수식 맥락): **문서 전반을 $\mathrm{prev\_low}_N$으로 통일**(다른 표기 사용 금지).
  - **제목/문장 속 기호:** “Donchian 이전 N-저가”처럼 **평문 N** 권장(수식은 수식 블록 안에서만).
- 수식 표기:
  - **디스플레이 수식:** `$$ ... $$` **만 사용**(`\[...\]` 전면 금지), **앞뒤에 빈 줄 1줄**씩 둔다.
  - **인라인 수식:** `$ ... $` 유지, **볼드/링크와는 반드시 공백 분리**.
  - **집합/연산자:** 수식 내부에서 `\{ \}`, `\le`, `\ge`, `\in` 등 LaTeX 명령 사용.
  - **절댓값:** `\lvert x \rvert` 권장(파이프 `|x|` 사용 금지).
  - **piecewise:** 리스트/표 내부일수록 **`$$` 블록 + 앞뒤 빈 줄**로 표기.
- 텍스트 변수명/컬럼명:
  - 언더스코어 포함 이름은 **코드스팬**으로 표기: `` `stop_level` ``, `` `Q_exec` ``.
  - **혼용 금지:** 코드스팬 안에 LaTeX 수식을 넣지 않는다(수식은 `$...$` 또는 `$$...$$`에서만).
  - **수식 내부 변수처럼 보이는 텍스트**는 `\texttt{stop\_level}` 또는 `\text{stop\_level}`로 표기(언더스코어는 `\_`).

---

## 코드 작성 시 주의점
1. 표준 라이브러리 **우선 사용**.
2. **코드 청소**: 동작·출력·공개 API는 유지한 채 미사용 import/변수/함수, 주석 처리된 코드, 임시 디버그 출력, 죽은/불필요 분기, 광범위 예외(pass) 제거.
3. **주석 가이드**: 공개 API 계약, 수식/타이밍 규약, 예외 처리 근거에 한해 **간결하게** 사용(중복·잡설 금지).

---

## 1) 데이터

### 역할
- **데이터 [수집 → 정합성 → 완전 조정 → 스냅샷 고정]** 만 담당한다.
- 전략용 **특징·의사결정(신호·스탑·사이징)** 로직은 포함하지 않는다.
- 출력은 **정합화된 시계열 스냅샷(DataFrame)** + **스냅샷 메타**로 한정한다.

### 수집
- 원천: 야후 / 트레이딩뷰 / 업비트 등 CSV·API.
- 인덱스: `DatetimeIndex[UTC]` 로 변환(타임존 명시), 시간순 정렬.
- 컬럼 통일: `open, high, low, close` (필수), 필요 시 컬럼명 매핑.

### 정합성
- 인덱스: **단조 증가**, **중복 없음**, **음수 시간 간격 금지**.
- 값의 범위: `open, high, low, close > 0`.
- 바 무결성: $L_t \le \min\{O_t, C_t, H_t\} \le H_t$ (매핑: $O \to \texttt{open}$, $H \to \texttt{high}$, $L \to \texttt{low}$, $C \to \texttt{close}$).
- 결측·중복 제거 & 거래 달력 정합(휴장·주말 필터, 가격 **보간 지양**).
- 위반 시 **즉시 실패**.

### 완전 조정
- 보정계수:

$$
a_t = \frac{\mathrm{AdjClose}_t}{\mathrm{Close}_t}
$$

- 조정 OHLC:

$$
X^{\mathrm{adj}}_t = X_t \cdot a_t,\quad X \in \{ O, H, L, C \}
$$

- 산출 컬럼: `open_adj, high_adj, low_adj, close_adj` — 존재하면 **항상 이 컬럼만 사용**(혼용 금지).

### 스냅샷 고정
- 저장 형식: **Parquet** (UTC 인덱스 유지).
- 메타: **SHA-256** 해시, 생성 시각, `source, symbol, timezone, start, end, interval, rows, columns, snapshot_path, snapshot_sha256`.
- **계약**: 이후 단계(전략/시뮬레이션)에서 **정렬·중복 제거·품질 게이트 재수행 금지**.

### 비용·체결 메타(참조)
- 수수료율, 슬리피지율, 호가/로트 단위, 포인트/핍 값은 **참조 상수 저장만** 수행(계산은 시뮬레이션 단계).

---

## 2) 전략

### 역할
- 데이터 스냅샷을 입력으로 **특징 생성 → 의사결정(신호·스탑) → 사이징**만 수행한다.
- 데이터 정합화/스냅샷 저장은 **데이터 단계 책임**.

### 입력 계약
- `*_adj` 존재 시 **항상 우선 사용**(`close_adj`, `low_adj` 등).

### 2.1 특징(피처)
**SMA**

$$
\mathrm{SMA}_{n}(t)=\frac{1}{n}\sum_{i=0}^{n-1} P_{t-i}
$$

$$
\mathrm{SMA}_{n}(t)=\mathrm{SMA}_{n}(t-1)+\frac{P_t-P_{t-n}}{n}
$$

- 입력: `close_adj`(우선), 없으면 `close`.
- 윈도 경계 전 구간은 `NaN`(의사결정 금지).

**Donchian 이전 N-저가 기준선**

$$
\mathrm{prev\_low}_N(t-1)=\min_{0\le i < N} L_{t-1-i}
$$

- 입력: `low_adj`(우선), 없으면 `low`.
- 초기 $t<N$ 구간은 `NaN`.

### 2.2 의사결정
**A) 신호 — SMA10/50 크로스(롱 온리)**  
- 결정: **Close** ($t-1$) 확정 특징.
- 체결: **Open** ($t$).
- 타이브레이크(안정 표기):

$$
\texttt{signal\_next}=
\begin{cases}
1 & \text{if } \mathrm{SMA}_{10}-\mathrm{SMA}_{50}>\epsilon \\
0 & \text{otherwise}
\end{cases}
\qquad (\epsilon\ge 0)
$$

**B) 스탑 — Donchian N-저가 이탈(롱 기준)**  
- 판정: **Close** ($t$) 에서 $L_t \le \mathrm{prev\_low}_N(t-1)$.
- 체결: **Open** ($t+1$).
- 산출: $\texttt{stop\_hit}\in\{\mathrm{True},\mathrm{False}\}$, $\texttt{stop\_level}=\mathrm{prev\_low}_N(t)$.

### 2.3 사이징 — Fixed Fractional(스펙 산출)
- 전략은 **룩어헤드 방지**를 위해 최종 수량을 계산하지 않고, 다음 **사이징 스펙**을 반환한다: 위험 비율 $f$, 스탑 레벨 `stop_level`, lot/호가 단위 $s$, 자산군 파라미터(예: 선물 승수 $V$, FX pip 값 $PV$ 등).
- 실제 위험 예산 $R=f\cdot E$, 스탑 거리 $D=\text{Entry}-\text{stop\_level}$, 수량 $Q$ 및 실행 수량 $Q_{\text{exec}}$ 계산은 **시뮬레이션 On-Open**에서 수행한다.

---

## 3) 시뮬레이션

### 역할
- 전략 출력(특징·신호·스탑·사이징 스펙)을 **타임라인 규약대로 집행**하여 **체결·비용 반영** 후 **산출물 고정**.
- 데이터 정합화/특징/의사결정 재구현 **금지**(전략 호출만).

### 타임라인(룩어헤드 금지, 우선순위)
- **신호**: **Close** ($t-1$) 결정 → **Open** ($t$) 체결.
- **스탑**: **Close** ($t$) 판정 → **Open** ($t+1$) 체결.
- 충돌 시 **스탑 > 신호**(청산 우선).

### 체결·비용·사이징 반영
- 체결가: **매수:** $\text{Open\_eff}(1+\text{slip})$, **매도:** $\text{Open\_eff}(1-\text{slip})$  
  정의: $\text{Open\_eff}=\text{open\_adj}$ (존재 시), 그 외 $\text{open}$.
- 수수료: $\lvert \text{체결가} \rvert \times \text{수량} \times \text{commission\_rate}$.
- **사이징 계산(On-Open)**:
  - 위험 예산: $R=f\cdot E$.
  - 스탑 거리(롱): $D=\text{Entry}-\text{stop\_level}\; (D\le 0 \Rightarrow Q=0)$.
  - 자산군별 수량:

$$Q_{\text{stock/coin}}=\left\lfloor \frac{R}{D} \right\rfloor$$

$$Q_{\text{fut}}=\left\lfloor \frac{R}{D\cdot V} \right\rfloor$$

$$Q_{\text{FX}}=\left\lfloor \frac{R}{D_{\mathrm{pips}}\cdot PV} \right\rfloor$$

  - 실행 수량(하향 라운딩): $Q_{\text{exec}}=\Big\lfloor \dfrac{Q}{s} \Big\rfloor \cdot s$.
  - $D\le 0$ 또는 lot/호가 단위 미만이면 **주문 생성 없음**.

### 루프(롱 온리·0/1 보유)
1. **On-Open**: 전일 예약 체결 집행(스탑 우선) → 슬리피지·수수료 반영 → 포지션/현금/자본 업데이트 → **거래 로그 기록**.
2. **On-Close**: 특징 갱신(SMA, $\mathrm{prev\_low}_N$) → 신호 예약·스탑 판정 → **자본 곡선 스냅샷**.

### 입력/출력 계약
- 입력 스냅샷: `DatetimeIndex[UTC]` 단조증가, 중복 없음; 필수 `open, high, low, close`, 권장 `open_adj, high_adj, low_adj, close_adj`(혼용 금지).
- 전략 호출 결과: 특징(`sma{short}`, `sma{long}`, $\mathrm{prev\_low}_N(t)$), 의사결정(`signal_next`, `stop_hit`, `stop_level`), **사이징 스펙**($f, s, V, PV$).
- 출력(표준화):  
  `trades`(체결 시각, 방향, 사유, 수량, 체결가, 수수료, 슬리피지, 실현 손익, 체결 후 자본/포지션 등)  
  `equity_curve`(시각, 자본, 포지션, 누적 수수료/슬리피지, 최대낙폭 등)  
  `metrics`(총수익, MDD, 승률·페이오프, 샤프 등)  
  `run_meta`(설정값, 스냅샷 해시/기간, 생성 시각·버전)

### 실패/예외
- 스냅샷 무결성 위반 시 **즉시 실패**.
- 전략 결과가 `NaN`(윈도 경계 미충족)인 시점은 해당 의사결정 **생성 없음**.

### 설정
- 모든 파라미터(`sma_short, sma_long, epsilon, N, f, lot_step, commission_rate, slip, point_value/pip_value, h_max` 등)는 **런 구성으로 주입**하고, `run_meta`에 기록한다.

---

## 4) 검증·리포팅
- **분할**: 표본내/외, 워크포워드(예: 3y train → 1y test 롤링).
- **민감도/스트레스**: $n$(SMA), $N$(Donchian), $f$(위험 비율), 수수료/슬리피지 상향.
- **리포팅/시각화**: 조합별 `metrics` 집계, 베스트/로버스트 구간(IQR·최악 포함), 자본곡선·히트맵 등.
- **원칙**: 검증은 **시뮬레이션 엔진 반복 호출만** 수행(전략/체결 로직 재구현 금지).

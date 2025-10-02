# 목표

* **데이터**[수집 → 정합성 → 완전 조정 → 기준 통화 환산·세션/캘린더 메타 → 스냅샷 고정] → **전략**[피처 → 신호·스탑 → 사이징 → 리밸런싱 스펙] → **시뮬레이션**[타임라인 집행 → 스탑 우선, 룩어헤드 금지 준수 → 리밸런싱 집행 → 비용 반영 → 산출물 고정] → **검증·리포팅**[데이터/스냅샷 무결성, 전략 규약 준수, 타임라인·회계 검증, 워크포워드, 민감도, 성과 시각화]

**기본 규약(요약)**

* SSOT 준수, `*_adj` 우선 사용, **Stop > Signal**, **룩어헤드 금지**.
* 기준 통화 평가 일원화(**항상 `base_currency` 사용**), **자산/호가/환율 메타**와 현금흐름·목표비중 입력은 **스냅샷/런 메타에 해시로 고정**: `instrument_registry_hash`, `cash_flow_source`, `target_weights_source`.

---

## 아키텍처 원칙(SSOT)

* 데이터 정합화는 **데이터 단계**에만, 특징 생성·의사결정은 **전략 단계**에만, 타임라인·체결·비용·우선순위는 **시뮬레이션 단계**에만 구현한다.
* **전략은 스펙 산출까지만** 수행한다(신호·스탑·사이징·리밸런싱 **스펙**). **시뮬레이션은 집행을 전담**한다(스탑/신호/리밸런싱 **집행**). 역할 경계를 문장으로 명확히 한다.
* 동일 규칙의 중복 구현을 금지한다.

---

## 표기 원칙(일관성)

* 용어: **특징(feature)**, **의사결정(decision)**, **스냅샷 메타(snapshot meta)**.

* 조정가: `*_adj` 컬럼이 존재하면 **항상 우선 사용**(혼용 금지).

* 기호 매핑: $O \to \texttt{open}$, $H \to \texttt{high}$, $L \to \texttt{low}$, $C \to \texttt{close}$, $\mathrm{AdjClose} \to \texttt{AdjClose}$.

* 시계열 표기:

  * 시점 $t$ 값은 하첨자(예: $L_t$).
  * **Donchian 표기 통일:** 본문·표·제목에서는 `prev_low_N`(평문), 수식 블록에서만 $\mathrm{prev_low}_N(\cdot)$ 사용.
  * **제목/문장 속 기호:** “Donchian 이전 N-저가”처럼 **평문 N** 권장(수식은 수식 블록 안에서만).

* 수식 표기:

  * **디스플레이 수식:** `$$ ... $$` **만 사용**(`\[...\]` 금지), **앞뒤에 빈 줄 1줄**을 **강제**한다.
  * **인라인 수식:** `$ ... $` 유지, **볼드/링크와는 반드시 공백 분리**.
  * **집합/연산자:** 수식 내부에서 `\{ \}`, `\le`, `\ge`, `\in` 등 LaTeX 명령 사용.
  * **절댓값:** **`\lvert \cdot \rvert` 사용**(파이프 `|x|` 사용 금지).
  * **piecewise:** 리스트/표 내부일수록 **`$$` 블록 + 앞뒤 빈 줄**로 표기.

* 텍스트 변수명/컬럼명:

  * 언더스코어 포함 이름은 **코드스팬**으로 표기: `` `stop_level` ``, `` `Q_exec` ``.
  * **혼용 금지:** 코드스팬 안에 LaTeX 수식을 넣지 않는다(수식은 `$...$` 또는 `$$...$$`에서만).
  * **수식 내부의 텍스트 변수 표기:** `\texttt{...}`만 사용(예: `\texttt{Open\_eff}`, `\texttt{stop\_level}`; 언더스코어는 `\_`).

---

## 코드 작성 시 주의점
> Python 버전: 3.11 이상 (python --version으로 확인)
1. 표준 라이브러리 **우선 사용**.
2. **코드 청소**: 동작·출력·공개 API는 유지한 채 미사용 import/변수/함수, 주석 처리된 코드, 임시 디버그 출력, 죽은/불필요 분기, 광범위 예외(pass) 제거.
3. **주석 가이드**: 공개 API 계약, 수식/타이밍 규약, 예외 처리 근거에 한해 **간결하게** 사용(중복·잡설 금지).

---

## 1) 데이터

### 데이터 수집

* 원천 API에서 OHLCV를 수집해 컬럼을 `open, high, low, close, volume`으로 표준화하고, 인덱스를 **UTC `DatetimeIndex`**로 정렬합니다.
* **기준 통화(`base_currency`)**를 명시하고, 비기준 통화 자산은 **동일 타임스탬프의 FX 시계열**을 함께 로드합니다.
* 종목별 거래 캘린더 메타의 명칭을 **`calendar_id`**로 고정합니다(예: `XNYS`, `XKOS`, `24x7`). 이후 **On-Open/On-Close** 타이밍 검증에 사용합니다.

### 데이터 정합성

* **인덱스 무결성**: `DatetimeIndex[UTC]` 단조증가, 중복 없음.

* **바 무결성**(분리 표기):

  $$
  L_t \le \min{O_t, C_t}
  $$

  $$
  \max{O_t, C_t} \le H_t
  $$

* **값 범위·결측**: `open, high, low, close` 양수, 필수 컬럼 결측 없음.

* **메타 검증 항목**: `lot_step > 0`, `price_step > 0`, `calendar_id` 키 일치, 해당 시점 **FX 레이트 존재**(부재 시 즉시 실패).

### 완전 조정

* 조정 계수 $a_t=\dfrac{\mathrm{AdjClose}_t}{\mathrm{Close}_t}$로 `open_adj, high_adj, low_adj, close_adj`를 생성합니다(미제공 시 $a_t=1.0$).
* 이후 단계에서 `*_adj`가 존재하면 **항상 우선 사용**합니다(혼용 금지).

### 기준 통화 환산·세션/캘린더 메타

* 가격 통화가 기준 통화와 다를 경우, **해당 시점 FX를 곱하거나(현지→기준 통화) 나누는 규약을 한 문장으로 명시**하여 평가·손익을 **기준 통화로 환산**합니다.
* 자산별 **`calendar_id`**를 스냅샷에 연결하고, 시뮬레이션의 타임라인 검증 기준으로 사용합니다.

### 스냅샷 고정

* 저장 형식: **Parquet**.
* **메타데이터**에는 다음을 포함합니다:
  `source, symbol, start, end, interval, rows, columns, snapshot_path, snapshot_sha256, collected_at, timezone, base_currency, fx_source, fx_source_ts, calendar_id, instrument_registry_hash`.
* 스냅샷 해시(`snapshot_sha256`)로 **재현성**을 고정합니다.

---

## 2) 전략

### 역할

* 데이터 스냅샷을 입력으로 **특징 생성 → 의사결정(신호·스탑) → 사이징 → 리밸런싱 스펙**만 수행한다.
* 데이터 정합화/스냅샷 저장은 **데이터 단계 책임**.

### 입력 계약

* `*_adj` 존재 시 **항상 우선 사용**(`close_adj`, `low_adj` 등, 혼용 금지).

### 2.1 특징(피처)

**SMA**

$$
\mathrm{SMA}*{n}(t)=\frac{1}{n}\sum*{i=0}^{n-1} P_{t-i}
$$

$$
\mathrm{SMA}*{n}(t)=\mathrm{SMA}*{n}(t-1)+\frac{P_t-P_{t-n}}{n}
$$

* 입력: `close_adj`(우선), 없으면 `close`.
* 윈도 경계 전 구간은 `NaN`(의사결정 금지).

**Donchian 이전 N-저가 기준선 (prev_low_N)**

$$
\mathrm{prev_low}*N(t-1)=\min*{0\le i < N} L_{t-1-i}
$$

* 입력: `low_adj`(우선), 없으면 `low`.
* 초기 $t<N$ 구간은 `NaN`.

### 2.2 의사결정

**A) 신호 — SMA10/50 크로스(롱 온리)**

* 결정: **Close** ($t-1$) 확정 특징.
* 체결: **Open** ($t$).

$$
\texttt{signal_next}=
\begin{cases}
1 & \text{if } \mathrm{SMA}*{10}-\mathrm{SMA}*{50}>\epsilon \
0 & \text{otherwise}
\end{cases}
\qquad (\epsilon\ge 0)
$$

**B) 스탑 — Donchian N-저가 이탈(롱 기준)**

* 판정: **Close** ($t$) 에서 $L_t \le \mathrm{prev_low}_N(t-1)$.
* 체결: **Open** ($t+1$).
* 산출: `stop_hit` $\in{\mathrm{True},\mathrm{False}}$, `stop_level` $=\mathrm{prev_low}_N(t)$.

### 2.3 사이징 — Fixed Fractional(스펙 산출)

* 전략은 **룩어헤드 방지**를 위해 최종 수량을 계산하지 않고, 다음 **사이징 스펙**을 반환한다: 위험 비율 $f$, 스탑 레벨 `stop_level`, 수량 라운딩 단위 $s$(**$s \equiv \texttt{lot_step}$**), 자산군 파라미터(예: 선물 승수 $V$, FX pip 값 $PV$ 등). 가격 라운딩은 **`price_step`**로 별도 처리한다.
* 실제 위험 예산 $R=f\cdot E$, 스탑 거리(롱) $D=\text{Entry}-\texttt{stop_level}$, 수량 $Q$ 및 실행 수량 $Q_{\text{exec}}$ 계산은 **시뮬레이션 On-Open**에서 수행한다.

$$
Q_{\text{stock/coin}}=\left\lfloor \frac{R}{D} \right\rfloor
$$

$$
Q_{\text{fut}}=\left\lfloor \frac{R}{D\cdot V} \right\rfloor
$$

$$
Q_{\text{FX}}=\left\lfloor \frac{R}{D_{\mathrm{pips}}\cdot PV} \right\rfloor
$$

* 실행 수량(하향 라운딩): $Q_{\text{exec}}=\Big\lfloor \dfrac{Q}{s} \Big\rfloor \cdot s$ (**$s \equiv \texttt{lot_step}$**).
* $D\le 0$ 또는 lot/호가 단위 미만이면 **주문 생성 없음**.

### 2.4 리밸런싱 — 현금흐름 리밸런싱(Cash-Flow Rebalancing)

**목표**

* 외부 **순현금흐름 $F_t$**(배당/이자/입출금)를 이용해 **불필요한 매도 없이** 목표 비중 $\mathbf{w}$에 **단순·일관되게** 근접한다.

**입력**

* 전일 평가액: 각 자산 $V_i(t!-!1)$, 총액 $P_{t-1}=\sum_i V_i(t!-!1)$
* 목표 비중: $\mathbf{w}=(w_1,\dots,w_n),\ \sum_i w_i=1$
* 순현금흐름: $F_t$ (**$F_t>0$ 유입**, **$F_t<0$ 유출**)

**기호 정의**

* $I^+={i:\ d_i>0}$, $I^-={i:\ d_i<0}$
* $S^+=\sum_{i\in I^+} d_i$, \quad $S^-=\sum_{i\in I^-} \lvert d_i\rvert$
* 갭 $d_i = T_i - V_i(t!-!1)$, \quad $T_i = w_i \cdot P^*$, \quad $P^* = P_{t-1}+F_t$

**권장 집행 규칙(결정적)**

* **유입($F_t\ge 0$)**: **미달 자산에만 매수**.

$$
\Delta_i^{\mathrm{buy}} =
\begin{cases}
\dfrac{d_i}{S^+}\cdot \min(F_t,\ S^+) & i\in I^+,, S^+>0[6pt]
0 & \text{otherwise}
\end{cases}
$$

* **잔여 처리(유입)**: $F_t > S^+$이면 **잔여는 현금 보유**(추가 매수 금지).
* **유출($F_t<0$)**: **과체중 자산에서만 매도**.

$$
\Delta_i^{\mathrm{sell}} =
\begin{cases}
\dfrac{\lvert d_i\rvert}{S^-}\cdot \min(\lvert F_t\rvert,\ S^-) & i\in I^-,, S^->0[8pt]
w_i\cdot \lvert F_t\rvert & \text{if } S^-=0 \ (\text{과체중 없음})
\end{cases}
$$

**출력(리밸런싱 스펙)**

* `rebalancing_spec`:

  * `target_weights`: $\mathbf{w}$
  * `cash_flow`: $F_t$
  * `buy_notional[i]`: $\Delta_i^{\mathrm{buy}}$
  * `sell_notional[i]`: $\Delta_i^{\mathrm{sell}}$

**집행(시뮬레이션 연계)**

* **On-Open**에 스탑/신호 예약 체결 후, 위 **권장 규칙**에 따라 현금 범위 내에서 집행한다. 체결가·수수료·라운딩은 기존 규약을 따른다.

---

## 3) 시뮬레이션

### 역할/타임라인

* 전략 출력(특징·신호·스탑·사이징 스펙·리밸런싱 스펙)을 **타임라인 규약**에 따라 집행하여 **체결·비용 반영** 후 **산출물 고정**한다. 데이터 정합화/특징/의사결정 재구현은 금지(전략 호출만).
* **표준 타임라인 규약(단일 정의)**:
  **신호**는 **Close**($t!-!1$)에서 결정해 **Open**($t$)에 체결, **스탑**은 **Close**($t$)에서 판정해 **Open**($t+1$)에 체결하며, 충돌 시 **스탑 > 신호**이고 **룩어헤드 금지**를 준수한다. (다른 섹션에서는 본 규약을 참조로만 언급)

### 집행 순서(On-Open)

1. 전일 예약 **스탑 체결**
2. 전일 예약 **신호 체결**
3. **현금흐름 $F_t$ 반영**(입·출금/배당/이자)
4. **리밸런싱 집행**(전략의 `rebalancing_spec` 사용) — 미달 자산 **비례 매수**, 과체중 자산 **비례 매도**, 잔여 현금은 보유. 리밸런싱 체결 기록은 **`reason='rebalance'`**로 고정
5. 수수료·슬리피지 반영 및 자본 업데이트

### 체결·비용·사이징 반영

* 체결가: **매수** $\texttt{Open_eff}(1+\texttt{slip})$, **매도** $\texttt{Open_eff}(1-\texttt{slip})$
  정의: $\texttt{Open_eff}=\texttt{open_adj}$ (존재 시), 그 외 `open`. **가격 라운딩은 `price_step` 적용**.
* 수수료: $\lvert \text{체결가} \rvert \times \text{수량} \times \texttt{commission_rate}$.
* **사이징 계산(On-Open)**
  위험 예산: $R=f\cdot E$, 스탑 거리(롱): $D=\text{Entry}-\texttt{stop_level}$.

$$
Q_{\text{stock/coin}}=\left\lfloor \frac{R}{D} \right\rfloor
\quad
Q_{\text{fut}}=\left\lfloor \frac{R}{D\cdot V} \right\rfloor
\quad
Q_{\text{FX}}=\left\lfloor \frac{R}{D_{\mathrm{pips}}\cdot PV} \right\rfloor
$$

* 실행 수량(하향 라운딩): $Q_{\text{exec}}=\Big\lfloor \dfrac{Q}{\texttt{lot_step}} \Big\rfloor \cdot \texttt{lot_step}$ (**수량 라운딩 = `lot_step`**, **가격 라운딩 = `price_step`**).
* **규칙:** **$D\le 0 ;\Rightarrow;$ 주문 없음**.

### 포트폴리오 모드

* 복수 종목 동시 보유를 기본으로 하며, 종목별 **수수료·슬리피지·`lot_step`/`price_step`**·자산군 파라미터(`V`, `PV`)를 적용한다.
* 리밸런싱 체결은 `reason='rebalance'`로 기록한다.

### 기준 통화 평가

* 자본 곡선은 **기준 통화(`base_currency`)**로 산출한다. 가격 통화가 다를 경우 시점별 FX로 환산 후 합산한다.

$$
E_t=\texttt{Cash}^{\text{base}}*t+\sum_i \texttt{Position}*{i,t}\cdot \Big(P_{i,t}^{\text{local}}\times \texttt{FX}_{i\to \text{base},t}\Big)
$$

### 루프

1. **On-Open**: 표준 순서(**스탑 → 신호 → 현금흐름 → 리밸런싱 → 비용 반영**)에 따라 집행 → 포지션/현금/자본 업데이트 → **거래 로그 기록**
2. **On-Close**: 특징 갱신(SMA, $\mathrm{prev_low}_N$) → 신호 예약·스탑 판정 → **자본 곡선 스냅샷**

### 입력/출력 계약

* 입력 스냅샷: `DatetimeIndex[UTC]` 단조증가·중복 없음; 필수 `open, high, low, close`, 권장 `open_adj, high_adj, low_adj, close_adj`(혼용 금지).
* 전략 결과: 특징(`sma{short}`, `sma{long}`, $\mathrm{prev_low}_N(t)$), 의사결정(`signal_next`, `stop_hit`, `stop_level`), **사이징 스펙**($f, s, V, PV$; 여기서 $s\equiv\texttt{lot_step}$), **리밸런싱 스펙**(**`target_weights`**, **`cash_flow`**, **`buy_notional`**, **`sell_notional`**).
* 출력(표준화):
  `trades`(시각, 심볼, 방향, 사유[signal/stop/rebalance], 수량, 체결가, 수수료, 슬리피지, 실현 손익, 체결 후 자본/포지션)
  `equity_curve`(**기준 통화 기준**, 시각, 자본, 포지션, 누적 수수료/슬리피지, 최대낙폭 등)
  `metrics`(**기준 통화 기준**, 총수익, MDD, 승률·페이오프, 샤프 등)
  `run_meta`(설정값, `base_currency`, 스냅샷 해시/기간, `instrument_registry_hash`, `cash_flow_source`, `target_weights_source`, 생성 시각·버전)

### 실패/예외

* **즉시 실패**: 스냅샷 무결성 위반, 타임라인 위반, 룩어헤드 사용.
* **경고 후 스킵**: `target_weights`/`cash_flow` 누락, 라운딩 불능 등(동일 문구를 데이터·시뮬레이션·검증 섹션에 적용).

### 설정

* 모든 파라미터(`sma_short, sma_long, epsilon, N, f, lot_step, price_step, commission_rate, slip, V, PV, base_currency`)는 **런 구성으로 주입**하고, `run_meta`에 기록한다.

---

## 4) 검증·리포팅

*요약 카드/게이트 기준: 워크포워드·민감도 및 게이트 기준은 본 섹션 하위 문서에 따름.*

* **데이터/스냅샷 무결성 검증**
  **의도:** 분석에 넣은 데이터가 깨끗하고 규칙대로 저장되어 **같은 입력이면 같은 결과가 다시 나오도록** 재현성을 보장합니다.

* **전략 규약 준수 검증**
  **의도:** 문서에 정의된 **타이밍/규칙(전날 종가로 결정, 다음날 시가로 체결 등)**을 그대로 따랐는지 확인합니다.

* **시뮬레이션 타임라인·회계 검증**
  **의도:** **주문·체결·수수료·슬리피지·수량 라운딩**이 규칙대로 계산되고, **현금+포지션=자본** 회계 등식이 맞는지 확인합니다.

* **일반화 신뢰도 평가**
  **의도:** 특정 구간의 **우연한 성과가 아닌지** 보기 위해 워크포워드로 구간을 나눠도 성능이 유지되는지, **비용/파라미터가 조금 변해도** 전략이 무너지지 않는지 점검합니다.

* **성과 시각화**
  **의도:** **자본곡선, 최대낙폭, 롤링 샤프, 거래별 손익 분포** 등을 보기 쉽게 그려 **성과와 위험을 한눈에** 파악합니다.

**검증 로그/출력 규격:** 시뮬레이션 산출물(`trades`, `equity_curve`, `metrics`, `run_meta`)과 **필드명 일치** 여부를 재확인합니다.

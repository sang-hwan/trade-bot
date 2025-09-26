"""
이 파일의 목적:
- Fixed Fractional(트레이드당 f%) 공식을 표준 라이브러리만으로 구현하여
  주식/코인·선물·FX의 실행 수량(Q_exec)을 계산합니다.

사용되는 변수와 함수 목록:
- 변수
  - 없음
- 함수
  - risk_budget(E: float, f: float) -> float
    - 입력값: E>0 — 계좌 자본, f>0 — 트레이드당 위험 비율(예: 0.01)
    - 출력값: float — 위험 예산 R=f·E
  - stop_distance_from_prev_low(entry: float, prev_low: float) -> float
    - 입력값: entry>0 — 엔트리 가격, prev_low>=0 — 전일 Donchian 최저 L_N(t-1)
    - 출력값: float — 스탑 거리 D=entry−prev_low (롱 기준)
  - floor_to_step(q: float, step: float) -> float
    - 입력값: q>=0 — 원시 수량, step>0 — 로트/수량 스텝
    - 출력값: float — 하향 보정된 수량
  - qty_stock_coin_ff(E: float, f: float, entry: float, prev_low: float, *, lot_step: float=1.0) -> float
    - 입력값: E,f>0; entry>prev_low; lot_step>0
    - 출력값: float — 주식/코인 실행 수량 Q_exec=⌊⌊R/D⌋/s⌋·s
  - qty_futures_ff(E: float, f: float, entry: float, prev_low: float, V: float, *, lot_step: float=1.0) -> float
    - 입력값: V>0 — 선물 포인트가치; 그 외 동일
    - 출력값: float — 선물 실행 수량 Q_exec=⌊⌊R/(D·V)⌋/s⌋·s
  - qty_fx_ff(E: float, f: float, D_pips: float, PV: float, *, lot_step: float=1.0) -> float
    - 입력값: D_pips>0 — 스탑 거리(핍), PV>0 — 핍가치
    - 출력값: float — FX 실행 수량 Q_exec=⌊⌊R/(D_pips·PV)⌋/s⌋·s

파일의 흐름(→):
- 위험 예산 R=f·E 계산 → 스탑 거리 D 산출 → 자산 유형별 원시 수량 Q 계산 → lot_step으로 하향 보정하여 Q_exec 반환
"""

import math

def risk_budget(E: float, f: float) -> float:
    if E <= 0 or f <= 0:
        raise ValueError("E>0 and f>0 required")
    return f * E

def stop_distance_from_prev_low(entry: float, prev_low: float) -> float:
    D = float(entry) - float(prev_low)
    if not (entry > 0 and prev_low >= 0):
        raise ValueError("entry>0 and prev_low>=0 required")
    if D <= 0:
        raise ValueError("entry must be greater than prev_low (D>0)")
    return D

def floor_to_step(q: float, step: float) -> float:
    if step <= 0:
        raise ValueError("step>0 required")
    return math.floor(float(q) / step) * step

def qty_stock_coin_ff(E: float, f: float, entry: float, prev_low: float, *, lot_step: float = 1.0) -> float:
    R = risk_budget(E, f)
    D = stop_distance_from_prev_low(entry, prev_low)
    Q = math.floor(R / D)
    return floor_to_step(Q, lot_step)

def qty_futures_ff(E: float, f: float, entry: float, prev_low: float, V: float, *, lot_step: float = 1.0) -> float:
    if V <= 0:
        raise ValueError("V>0 required")
    R = risk_budget(E, f)
    D = stop_distance_from_prev_low(entry, prev_low)
    Q = math.floor(R / (D * V))
    return floor_to_step(Q, lot_step)

def qty_fx_ff(E: float, f: float, D_pips: float, PV: float, *, lot_step: float = 1.0) -> float:
    if E <= 0 or f <= 0 or D_pips <= 0 or PV <= 0:
        raise ValueError("E,f,D_pips,PV must be > 0")
    R = risk_budget(E, f)
    Q = math.floor(R / (float(D_pips) * float(PV)))
    return floor_to_step(Q, lot_step)

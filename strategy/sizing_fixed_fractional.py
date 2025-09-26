"""
Fixed Fractional(f%)와 스탑 거리를 이용해 자산별 실행 수량(Q_exec)을 보수적으로 산출하는 순수 유틸리티

이 파일의 목적:
- 계좌 자본 E와 위험 비율 f로 위험 예산 R=f·E를 만들고, 엔트리 대비 전일 저점(또는 지정 저점)까지의 스탑 거리 D를 산출합니다.
- 자산 유형(주식/코인, 선물, FX)에 맞는 가치 스케일(포인트가치 V, 핍가치 PV)을 반영해 원시 수량을 계산한 뒤,
  거래소 규격 lot_step의 배수로 내림 보정하여 과체결을 방지합니다.

사용되는 변수와 함수 목록:
- 변수
  - 없음

- 함수
  - risk_budget(E: float, f: float)
    - 역할: 계좌 자본과 위험 비율로 위험 예산 R을 계산
    - 입력값: E: float - 계좌 자본(>0, 통화 단위), f: float - 트레이드당 위험 비율(>0, 예: 0.01=1%)
    - 반환값: float - 위험 예산 R=f·E

  - stop_distance_from_prev_low(entry: float, prev_low: float)
    - 역할: 롱 기준 스탑 거리 D=entry−prev_low 계산(L_N(t−1) 등 전일/이전 저점 사용)
    - 입력값: entry: float - 엔트리 가격(>0), prev_low: float - 전/이전 저점(≥0)
    - 반환값: float - 스탑 거리 D(>0)

  - floor_to_step(q: float, step: float)
    - 역할: 원시 수량 q를 lot_step 배수로 내림 보정(⌊q/step⌋·step)
    - 입력값: q: float - 원시 수량(≥0 권장), step: float - 수량 스텝(>0, 예: 1, 0.1, 0.001)
    - 반환값: float - 보정된 수량( step 배수 )

  - qty_stock_coin_ff(E: float, f: float, entry: float, prev_low: float, *, lot_step: float = 1.0)
    - 역할: 주식/코인에 대해 R과 D로 원시 수량 ⌊R/D⌋을 구하고 lot_step으로 내림 보정
    - 입력값: E: float - 계좌 자본(>0), f: float - 위험 비율(>0), entry: float - 엔트리(>0), prev_low: float - 전/이전 저점(≥0), lot_step: float = 1.0 - 수량 스텝(>0)
    - 반환값: float - 실행 수량 Q_exec=⌊⌊R/D⌋/lot_step⌋·lot_step

  - qty_futures_ff(E: float, f: float, entry: float, prev_low: float, V: float, *, lot_step: float = 1.0)
    - 역할: 선물 포인트가치 V(1포인트당 손익)를 반영해 계약 수를 보수적으로 산출
    - 입력값: E: float - 계좌 자본(>0), f: float - 위험 비율(>0), entry: float - 엔트리(>0), prev_low: float - 전/이전 저점(≥0), V: float - 포인트가치(>0), lot_step: float = 1.0 - 계약 스텝(>0)
    - 반환값: float - 실행 수량 Q_exec=⌊⌊R/(D·V)⌋/lot_step⌋·lot_step

  - qty_fx_ff(E: float, f: float, D_pips: float, PV: float, *, lot_step: float = 1.0)
    - 역할: FX에서 스탑 거리(핍) D_pips와 1로트 기준 핍가치 PV로 로트 수를 산출
    - 입력값: E: float - 계좌 자본(>0), f: float - 위험 비율(>0), D_pips: float - 스탑 거리(>0, pips), PV: float - 1로트 기준 1핍가치(>0, 통화/핍), lot_step: float = 1.0 - 로트 스텝(>0)
    - 반환값: float - 실행 로트 Q_exec=⌊⌊R/(D_pips·PV)⌋/lot_step⌋·lot_step

파일의 흐름(→ / ->):
- 위험 예산 R 산출 -> 스탑 거리 D 계산 -> 자산별 원시 수량 Q 계산(보수적 내림) -> lot_step 배수로 최종 하향 보정 -> Q_exec 반환
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
    Q_raw = R / D
    return floor_to_step(Q_raw, lot_step)

def qty_futures_ff(E: float, f: float, entry: float, prev_low: float, V: float, *, lot_step: float = 1.0) -> float:
    if V <= 0:
        raise ValueError("V>0 required")
    R = risk_budget(E, f)
    D = stop_distance_from_prev_low(entry, prev_low)
    Q_raw = R / (D * V)
    return floor_to_step(Q_raw, lot_step)

def qty_fx_ff(E: float, f: float, D_pips: float, PV: float, *, lot_step: float = 1.0) -> float:
    if E <= 0 or f <= 0 or D_pips <= 0 or PV <= 0:
        raise ValueError("E,f,D_pips,PV must be > 0")
    R = risk_budget(E, f)
    Q_raw = R / (float(D_pips) * float(PV))
    return floor_to_step(Q_raw, lot_step)

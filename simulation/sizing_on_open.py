# simulation/sizing_on_open.py
"""
사이징 계산(온-오픈 전용, 룩어헤드 금지).
- 입력: entry_price(슬리피지 반영가), 직전 종가 기준 equity, 스펙(f, stop_level, lot_step[, V|PV])
- 규칙: R=f·E, D=Entry-stop_level(롱), Q_raw=R/denom, Q_exec=⌊Q_raw/lot_step⌋·lot_step
       denom = D(현물) | D·V(선물) | D_pips·PV(FX)
- 계산만 수행(주문/체결/수수료 반영 없음).
"""

from __future__ import annotations

from math import floor, isfinite

__all__ = ["calc_distance_long", "floor_step", "size_from_spec"]


def calc_distance_long(entry_price: float, stop_level: float) -> float:
    """롱 기준 스탑 거리: D = max(entry - stop_level, 0.0). 비정상 값은 0.0."""
    try:
        D = float(entry_price) - float(stop_level)
    except (TypeError, ValueError):
        return 0.0
    if not isfinite(D) or D <= 0.0:
        return 0.0
    return D


def floor_step(q: float, step: float) -> float:
    """로트/호가 단위 하향 라운딩: ⌊q/step⌋·step (step<=0 또는 q<=0이면 0.0)."""
    try:
        qf = float(q)
        sf = float(step)
    except (TypeError, ValueError):
        return 0.0
    if sf <= 0.0 or qf <= 0.0 or (not isfinite(qf)) or (not isfinite(sf)):
        return 0.0
    return floor(qf / sf) * sf


def _denominator(D: float, V: float | None, PV: float | None) -> float:
    """분모: 선물 D·V > FX D·PV > 현물 D. D 비정상이면 0.0."""
    if not isfinite(D) or D <= 0.0:
        return 0.0
    if V is not None:
        return D * float(V)
    if PV is not None:
        return D * float(PV)
    return D


def size_from_spec(
    entry_price: float,
    equity: float,
    *,
    f: float,
    stop_level: float,
    lot_step: float,
    V: float | None = None,
    PV: float | None = None,
) -> dict[str, float]:
    """
    Fixed Fractional 스펙으로부터 온-오픈 수량 산출(계산만).
    반환: {"Q_raw","Q_exec","R","D","denom","lot_step"}
    """
    try:
        E = float(equity)
        fval = float(f)
    except (TypeError, ValueError):
        return {"Q_raw": 0.0, "Q_exec": 0.0, "R": 0.0, "D": 0.0, "denom": 0.0, "lot_step": 0.0}

    # lot_step 비정상(NaN/inf/<=0) 방어: 출력에도 0.0을 기록해 JSON/후속 계산 안정성 확보
    try:
        lot_step_f = float(lot_step)
    except (TypeError, ValueError):
        lot_step_f = 0.0
    lot_step_f = lot_step_f if (isfinite(lot_step_f) and lot_step_f > 0.0) else 0.0

    R = fval * E
    D = calc_distance_long(entry_price, stop_level)
    denom = _denominator(D, V, PV)

    if R <= 0.0 or denom <= 0.0 or not (isfinite(R) and isfinite(denom)):
        return {"Q_raw": 0.0, "Q_exec": 0.0, "R": float(R), "D": float(D), "denom": float(denom), "lot_step": lot_step_f}

    Q_raw = R / denom
    Q_exec = floor_step(Q_raw, lot_step_f)

    return {
        "Q_raw": float(Q_raw),
        "Q_exec": float(Q_exec),
        "R": float(R),
        "D": float(D),
        "denom": float(denom),
        "lot_step": lot_step_f,
    }

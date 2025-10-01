# sizing_on_open.py
"""
사이징 계산(온-오픈 전용, 룩어헤드 금지).
- 입력: entry_price(슬리피지 반영가), 직전 종가 기준 equity, 스펙(f, stop_level, lot_step[, V|PV])
- 규칙: R=f·E, D=Entry-stop_level(롱), Q_raw=R/denom, Q_exec=⌊Q_raw/lot_step⌋·lot_step
       denom = D(주식/코인) | D·V(선물) | D_pips·PV(FX; D가 pips 단위로 주어졌다는 전제)
- 이 모듈은 계산만 수행(주문/체결/수수료 반영 없음).
"""

from __future__ import annotations

from math import floor, isfinite

__all__ = ["calc_distance_long", "floor_step", "size_from_spec"]


def calc_distance_long(entry_price: float, stop_level: float) -> float:
    """롱 기준 스탑 거리: D = Entry - stop_level (D<=0이면 0.0)."""
    D = float(entry_price) - float(stop_level)
    return D if D > 0.0 else 0.0


def floor_step(q: float, step: float) -> float:
    """로트/호가 단위 하향 라운딩: ⌊q/step⌋·step (step<=0 또는 q<=0이면 0.0)."""
    q = float(q)
    step = float(step)
    if step <= 0.0 or q <= 0.0:
        return 0.0
    return floor(q / step) * step


def _denominator(D: float, V: float | None, PV: float | None) -> float:
    """
    분모 계산:
      - 선물: D·V(V가 주어지면 우선)
      - FX  : D_pips·PV(V가 없고 PV가 주어졌다고 가정)
      - 현물(주식/코인): D
    """
    if D <= 0.0:
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
    - R = f·equity
    - D = calc_distance_long(entry_price, stop_level)
    - denom = _denominator(D, V, PV)
    - Q_raw = R / denom (분모<=0 또는 비정상 값이면 0.0)
    - Q_exec = floor_step(Q_raw, lot_step)
    반환: {"Q_raw", "Q_exec", "R", "D", "denom", "lot_step"}
    """
    E = float(equity)
    f = float(f)
    R = f * E
    D = calc_distance_long(entry_price, stop_level)
    denom = _denominator(D, V, PV)

    if R <= 0.0 or denom <= 0.0 or not (isfinite(R) and isfinite(denom)):
        return {
            "Q_raw": 0.0,
            "Q_exec": 0.0,
            "R": R,
            "D": D,
            "denom": denom,
            "lot_step": float(lot_step),
        }

    Q_raw = R / denom
    Q_exec = floor_step(Q_raw, lot_step)

    return {
        "Q_raw": float(Q_raw),
        "Q_exec": float(Q_exec),
        "R": float(R),
        "D": float(D),
        "denom": float(denom),
        "lot_step": float(lot_step),
    }

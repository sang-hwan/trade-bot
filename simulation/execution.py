# execution.py
"""
체결 계산 유틸리티.
- 가격 컬럼: *_adj 우선 사용(혼용 금지)
- 타이밍: On-Open에서 슬리피지·수수료를 적용해 체결가/현금 변동 계산
- 가격 라운딩: price_step 제공 시 체결가 산출 직전 틱 라운딩 적용

공개 API
- round_price(price, price_step, *, mode="nearest_floor") -> float | None
- open_eff(row, *, slip: float, side: str, price_step: float | None = None, round_mode: str = "nearest_floor") -> float | None
- close_eff(row, *, price_step: float | None = None, round_mode: str = "nearest_floor") -> float | None
- calc_commission(price: float, qty: float, commission_rate: float) -> float
- apply_buy(cash: float, entry_price: float, qty: float, commission_rate: float) -> tuple[float, float, float]
- apply_sell(cash: float, avg_entry: float, exit_price: float, qty: float, commission_rate: float) -> tuple[float, float, float]
"""

from __future__ import annotations

# ── 표준 라이브러리 우선
from collections.abc import Iterable
from math import floor
from typing import Literal

# ── 서드파티
import pandas as pd

__all__ = [
    "round_price",
    "open_eff",
    "close_eff",
    "calc_commission",
    "apply_buy",
    "apply_sell",
]


# ---------- 내부 공통 ----------
def _pick_name(columns: Iterable[str], base: str) -> str:
    """컬럼 집합에서 *_adj 우선 선택."""
    adj = f"{base}_adj"
    return adj if adj in columns else base


# ---------- 가격 라운딩 ----------
def round_price(
    price: float,
    price_step: float,
    *,
    mode: Literal["nearest_floor", "floor"] = "nearest_floor",
    _tol: float = 1e-12,
) -> float | None:
    """
    틱 라운딩 유틸.

    규약:
      - mode="nearest_floor": 가장 가까운 틱으로 라운딩. 정확히 0.5틱 동률이면 **내림**(floor) 선택.
      - mode="floor": 항상 **내림**.
      - price_step<=0 또는 price<=0 인 경우 **라운딩 불능** → None 반환.

    반환:
      - 라운딩된 가격(float) 또는 None(주문 불가/라운딩 불능)
    """
    p = float(price)
    s = float(price_step)
    if not (p > 0.0 and s > 0.0):
        return None

    ticks = p / s
    t_floor = floor(ticks)
    frac = ticks - t_floor

    if mode == "floor":
        t = t_floor
    else:  # nearest_floor
        if frac > 0.5 + _tol:
            t = t_floor + 1
        elif frac < 0.5 - _tol:
            t = t_floor
        else:  # 동률(≈0.5) → 내림
            t = t_floor

    return float(t) * s


# ---------- 체결가 ----------
def open_eff(
    row: pd.Series,
    *,
    slip: float,
    side: str,
    price_step: float | None = None,
    round_mode: Literal["nearest_floor", "floor"] = "nearest_floor",
) -> float | None:
    """
    시가 효과가(슬리피지 반영). side ∈ {'buy','sell'}.
    - price_step가 주어지면 체결 직전 round_price() 적용.
    - 라운딩 불능(None)이면 주문 불가로 간주하고 None 반환.
    """
    base = float(row[_pick_name(row.index, "open")])
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    eff = base * (1.0 + float(slip)) if side == "buy" else base * (1.0 - float(slip))
    if price_step is None:
        return eff
    return round_price(eff, price_step, mode=round_mode)


def close_eff(
    row: pd.Series,
    *,
    price_step: float | None = None,
    round_mode: Literal["nearest_floor", "floor"] = "nearest_floor",
) -> float | None:
    """
    종가(조정가 우선).
    - 평가용으로 보통 라운딩이 필요 없지만, 필요 시 price_step 제공 가능.
    """
    base = float(row[_pick_name(row.index, "close")])
    if price_step is None:
        return base
    return round_price(base, float(price_step), mode=round_mode)


# ---------- 수수료 ----------
def calc_commission(price: float, qty: float, commission_rate: float) -> float:
    """수수료 = |price| × qty × commission_rate."""
    return abs(float(price)) * float(qty) * float(commission_rate)


# ---------- 현금/손익 반영 ----------
def apply_buy(cash: float, entry_price: float, qty: float, commission_rate: float) -> tuple[float, float, float]:
    """매수 집행 후 현금/수수료/총원가 반환: (cash_after, commission, total_cost)."""
    commission = calc_commission(entry_price, qty, commission_rate)
    total_cost = float(entry_price) * float(qty) + commission
    return float(cash) - total_cost, commission, total_cost


def apply_sell(
    cash: float,
    avg_entry: float,
    exit_price: float,
    qty: float,
    commission_rate: float,
) -> tuple[float, float, float]:
    """매도 집행 후 현금/수수료/실현손익 반환: (cash_after, commission, realized_pnl)."""
    commission = calc_commission(exit_price, qty, commission_rate)
    realized = (float(exit_price) - float(avg_entry)) * float(qty) - commission
    cash_after = float(cash) + float(exit_price) * float(qty) - commission
    return cash_after, commission, realized

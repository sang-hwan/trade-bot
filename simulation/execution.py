# execution.py
"""
체결 계산 유틸리티 (Python 3.11+).
- 가격 컬럼: *_adj 우선 사용(혼용 금지)
- 타이밍: On-Open에서 슬리피지 적용, 수수료는 별도 헬퍼 사용
- 라운딩(검증 게이트와 일치):
  * open_eff: 최근접(동률=올림, ROUND_HALF_UP)
  * close_eff: 필요 시 최근접(동률=올림)

공개 API
- round_price(price, price_step, *, mode="nearest") -> float | None
- open_eff(row, *, slip: float, side: str, price_step: float | None = None, round_mode: str = "nearest") -> float | None
- close_eff(row, *, price_step: float | None = None, round_mode: str = "nearest") -> float | None
- calc_commission(price: float, qty: float, commission_rate: float) -> float
- apply_buy(cash: float, entry_price: float, qty: float, commission_rate: float) -> tuple[float, float, float]
- apply_sell(cash: float, avg_entry: float, exit_price: float, qty: float, commission_rate: float) -> tuple[float, float, float]
"""

from __future__ import annotations

# 표준 라이브러리 (우선 사용)
from collections.abc import Iterable
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP, getcontext
from typing import Literal

# 서드파티
import pandas as pd

__all__ = (
    "round_price",
    "open_eff",
    "close_eff",
    "calc_commission",
    "apply_buy",
    "apply_sell",
)

# Decimal 정밀도(틱 경계 안전)
getcontext().prec = 28


# ---------- 내부 공통 ----------
def _pick_name(columns: Iterable[str], base: str) -> str:
    """컬럼 집합에서 *_adj 우선 선택."""
    adj = f"{base}_adj"
    return adj if adj in columns else base


def _round_nearest_half_up_dec(val_dec: Decimal, step_dec: Decimal) -> Decimal:
    """step 단위 최근접 라운딩(동률=올림; ROUND_HALF_UP)."""
    if step_dec <= 0:
        raise ValueError("price_step must be > 0")
    q = (val_dec / step_dec).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return q * step_dec


# ---------- 가격 라운딩(범용) ----------
def round_price(
    price: float,
    price_step: float,
    *,
    mode: Literal["nearest", "floor"] = "nearest",
) -> float | None:
    """
    틱 라운딩(Decimal 기반).
    - mode="nearest": 최근접(동률=올림)
    - mode="floor": 내림
    - price_step<=0 또는 price<=0 → None
    """
    if mode not in ("nearest", "floor"):
        raise ValueError("mode must be 'nearest' or 'floor'")

    p = Decimal(str(price))
    s = Decimal(str(price_step))
    if not (p > 0 and s > 0):
        return None

    if mode == "floor":
        t = (p / s).to_integral_value(rounding=ROUND_FLOOR)
        return float(t * s)

    return float(_round_nearest_half_up_dec(p, s))


# ---------- 체결가 ----------
def open_eff(
    row: pd.Series,
    *,
    slip: float,
    side: str,
    price_step: float | None = None,
    round_mode: Literal["nearest", "floor"] = "nearest",
) -> float | None:
    """
    시가 효과가(슬리피지 반영). side ∈ {'buy','sell'}.
    - 라운딩: 최근접(동률=올림) 또는 내림
    - 전과정 Decimal 계산(슬리피지 곱셈 → 틱 라운딩)
    - price_step 미지정 시 라운딩 생략; 결과 ≤0이면 None
    """
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    if round_mode not in ("nearest", "floor"):
        raise ValueError("round_mode must be 'nearest' or 'floor'")

    base_dec = Decimal(str(row[_pick_name(row.index, "open")]))
    s_dec = Decimal(str(slip))
    one = Decimal("1")
    mult = (one + s_dec) if side == "buy" else (one - s_dec)
    eff_dec = base_dec * mult

    if price_step is None:
        px_dec = eff_dec
    else:
        step_dec = Decimal(str(price_step))
        if round_mode == "floor":
            t = (eff_dec / step_dec).to_integral_value(rounding=ROUND_FLOOR)
            px_dec = t * step_dec
        else:
            px_dec = _round_nearest_half_up_dec(eff_dec, step_dec)

    return float(px_dec) if px_dec > 0 else None


def close_eff(
    row: pd.Series,
    *,
    price_step: float | None = None,
    round_mode: Literal["nearest", "floor"] = "nearest",
) -> float | None:
    """
    종가(조정가 우선). 평가용.
    - 보통 라운딩 불필요. 필요 시 round_price 사용.
    """
    base = float(row[_pick_name(row.index, "close")])
    return base if price_step is None else round_price(base, float(price_step), mode=round_mode)


# ---------- 수수료 ----------
def calc_commission(price: float, qty: float, commission_rate: float) -> float:
    """수수료 = |price| × qty × commission_rate."""
    return abs(float(price)) * float(qty) * float(commission_rate)


# ---------- 현금/손익 반영 ----------
def apply_buy(cash: float, entry_price: float, qty: float, commission_rate: float) -> tuple[float, float, float]:
    """매수 후 (현금, 수수료, 총원가)."""
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
    """매도 후 (현금, 수수료, 실현손익)."""
    commission = calc_commission(exit_price, qty, commission_rate)
    realized = (float(exit_price) - float(avg_entry)) * float(qty) - commission
    cash_after = float(cash) + float(exit_price) * float(qty) - commission
    return cash_after, commission, realized

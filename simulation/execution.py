# execution.py
"""
체결 계산 유틸리티.
- 가격 컬럼: *_adj 우선 사용(혼용 금지)
- 타이밍: On-Open에서 슬리피지·수수료를 적용해 체결가/현금 변동 계산

공개 API
- open_eff(row, *, slip: float, side: str) -> float
- close_eff(row) -> float
- calc_commission(price: float, qty: float, commission_rate: float) -> float
- apply_buy(cash: float, entry_price: float, qty: float, commission_rate: float) -> tuple[float, float, float]
- apply_sell(cash: float, avg_entry: float, exit_price: float, qty: float, commission_rate: float) -> tuple[float, float, float]
"""

from __future__ import annotations

from collections.abc import Iterable
import pandas as pd

__all__ = [
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


# ---------- 체결가 ----------
def open_eff(row: pd.Series, *, slip: float, side: str) -> float:
    """시가 효과가(슬리피지 반영). side: 'buy' | 'sell'."""
    base = float(row[_pick_name(row.index, "open")])
    return base * (1.0 + float(slip)) if side == "buy" else base * (1.0 - float(slip))


def close_eff(row: pd.Series) -> float:
    """종가(조정가 우선)."""
    return float(row[_pick_name(row.index, "close")])


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

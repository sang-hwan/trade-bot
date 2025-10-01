# engine.py
"""
백테스트 엔진(롱 온리·0/1 보유).
- 타임라인: On-Open 집행(스탑 우선) → On-Close 판정
- 룩어헤드 금지: 신호는 Close(t-1) 결정→Open(t) 집행, 스탑은 Close(t) 판정→Open(t+1) 집행
- SSOT: 전략(신호/스탑/스펙) 호출만, 집행/수수료/사이징 계산은 전용 유틸 호출만.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

# 전략: 피처/신호/스탑/사이징 스펙
from strategy.signals import sma_cross_long_only
from strategy.stops import donchian_stop_long
from strategy.sizing_spec import build_fixed_fractional_spec

# 시뮬레이션 유틸: 체결/수수료, 온-오픈 사이징
from simulation.execution import open_eff, close_eff, calc_commission, apply_buy, apply_sell
from simulation.sizing_on_open import size_from_spec


# ---------- 내부 검증 ----------
def _validate_snapshot(df: pd.DataFrame) -> None:
    """스냅샷 불변 규약 검증(정렬·수정 금지, 위반 시 즉시 실패)."""
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("index must be DatetimeIndex")
    if idx.tz is None or str(idx.tz) != "UTC":
        raise ValueError("index timezone must be UTC")
    if not idx.is_monotonic_increasing:
        raise ValueError("index must be strictly increasing (no reordering)")
    if idx.has_duplicates:
        raise ValueError("index must not contain duplicates")


# ---------- 로그 레코드 ----------
@dataclass
class Trade:
    ts: pd.Timestamp
    side: str      # 'buy' | 'sell'
    reason: str    # 'signal' | 'stop'
    qty: float
    price: float
    commission: float
    slip: float
    realized_pnl: float
    equity_after: float
    position_after: float


# ---------- 엔진 ----------
def run(
    df: pd.DataFrame,
    *,
    f: float,
    N: int,
    epsilon: float = 0.0,
    lot_step: float = 1.0,
    commission_rate: float = 0.0,
    slip: float = 0.0,
    V: Optional[float] = None,
    PV: Optional[float] = None,
    initial_equity: float = 1_000_000.0,
) -> Dict[str, Any]:
    """
    공개 API
    - 입력: DatetimeIndex[UTC] 스냅샷(변경 금지)
    - 출력: {'trades': DataFrame, 'equity_curve': DataFrame, 'metrics': dict, 'run_meta': dict}
    """
    if df.empty:
        raise ValueError("empty dataframe")
    _validate_snapshot(df)
    df = df.copy()  # 불변 보장(내용 수정 없음)

    # 전략 산출물(전략 모듈 호출만; 재구현 금지)
    signal_next = sma_cross_long_only(df, short=10, long=50, epsilon=epsilon)
    stop_df = donchian_stop_long(df, N=N)
    sizing_spec = build_fixed_fractional_spec(df, N=N, f=f, lot_step=lot_step, V=V, PV=PV)

    # 상태
    cash = float(initial_equity)
    qty = 0.0
    avg_entry = 0.0
    equity = float(initial_equity)
    pending_exit = False

    # 로그
    trades: list[Trade] = []
    eq_rows: list[dict[str, Any]] = []

    # MDD
    peak = equity
    mdd = 0.0

    for ts, row in df.iterrows():
        # ===== On-Open: 집행(스탑 우선) =====
        if pending_exit and qty > 0.0:
            sell_qty = qty
            exit_price = open_eff(row, slip=slip, side="sell")
            cash, commission, realized = apply_sell(cash, avg_entry, exit_price, sell_qty, commission_rate)
            qty = 0.0
            avg_entry = 0.0
            trades.append(Trade(ts, "sell", "stop", sell_qty, exit_price, commission, slip, realized, cash, qty))
        pending_exit = False

        # 진입(포지션 없고, signal_next[t]==1)
        if qty == 0.0:
            sig = signal_next.get(ts)
            if pd.notna(sig) and int(sig) == 1 and ts in sizing_spec.index:
                spec = sizing_spec.loc[ts]
                if pd.notna(spec.get("stop_level", pd.NA)):
                    entry_price = open_eff(row, slip=slip, side="buy")
                    sized = size_from_spec(
                        entry_price,
                        equity,  # 직전 Close 기준 자본
                        f=float(spec["f"]),
                        stop_level=float(spec["stop_level"]),
                        lot_step=float(spec["lot_step"]),
                        V=None if pd.isna(spec.get("V", pd.NA)) else float(spec["V"]),
                        PV=None if pd.isna(spec.get("PV", pd.NA)) else float(spec["PV"]),
                    )
                    Q_exec = float(sized["Q_exec"])
                    if Q_exec > 0.0:
                        # 현금 확인 후 집행
                        commission = calc_commission(entry_price, Q_exec, commission_rate)
                        total_cost = entry_price * Q_exec + commission
                        if total_cost <= cash:
                            cash, commission, total_cost = apply_buy(cash, entry_price, Q_exec, commission_rate)
                            qty = Q_exec
                            avg_entry = entry_price
                            trades.append(
                                Trade(ts, "buy", "signal", Q_exec, entry_price, commission, slip, 0.0, cash + qty * entry_price, qty)
                            )

        # ===== On-Close: 판정/스냅샷 =====
        close_px = close_eff(row)
        equity = cash + qty * close_px

        sh = stop_df["stop_hit"].get(ts)
        if pd.notna(sh) and bool(sh) and qty > 0.0:
            pending_exit = True

        peak = max(peak, equity)
        dd = 0.0 if peak == 0.0 else (equity - peak) / peak
        mdd = min(mdd, dd)

        eq_rows.append(
            {"ts": ts, "equity": equity, "position": qty, "cash": cash, "close": close_px,
             "pending_exit_next_open": pending_exit, "peak": peak, "drawdown": dd}
        )

    # ---- 출력 고정 ----
    trades_df = (
        pd.DataFrame([t.__dict__ for t in trades]).sort_values("ts").reset_index(drop=True) if trades else pd.DataFrame()
    )
    equity_curve = pd.DataFrame(eq_rows).set_index("ts") if eq_rows else pd.DataFrame()

    total_return = 0.0 if equity_curve.empty else (float(equity_curve["equity"].iloc[-1]) / initial_equity - 1.0)
    metrics = {"total_return": float(total_return), "mdd": float(mdd), "trades": int(len(trades_df))}
    run_meta = {
        "f": f,
        "N": N,
        "epsilon": epsilon,
        "lot_step": lot_step,
        "commission_rate": commission_rate,
        "slip": slip,
        "V": V,
        "PV": PV,
        "initial_equity": initial_equity,
        "version": "engine.v1.3",
    }

    return {"trades": trades_df, "equity_curve": equity_curve, "metrics": metrics, "run_meta": run_meta}

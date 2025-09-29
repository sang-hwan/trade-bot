from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from math import floor
from datetime import datetime

import pandas as pd


@dataclass
class ExecConfig:
    n_fast: int = 10
    n_slow: int = 50
    epsilon: float = 0.0
    donchian_n: int = 20
    f: float = 0.01
    lot_step: float = 1.0
    commission: float = 0.0005
    slippage: float = 0.0005
    asset_type: str = "price"
    point_value: float = 1.0
    initial_equity: float = 1_000_000.0
    use_adjusted: bool = True
    name: Optional[str] = None
    h_max: Optional[float] = None


def _col(df: pd.DataFrame, base: str, use_adj: bool) -> str:
    a = f"{base}_adj"
    if use_adj and a in df.columns:
        return a
    return base


def _floor_step(q: float, step: float) -> float:
    if step <= 0:
        return 0.0
    return floor(q / step) * step


def _risk_qty_price(R: float, entry: float, stop: float, step: float) -> float:
    D = entry - stop
    if entry <= 0 or stop < 0 or D <= 0 or R <= 0:
        return 0.0
    q = R / D
    return _floor_step(q, step)


def _fill_price(open_px: float, slippage: float, side: str) -> float:
    if side == "buy":
        return open_px * (1.0 + slippage)
    return open_px * (1.0 - slippage)


def _commission_cost(fill_px: float, qty: float, rate: float) -> float:
    if qty <= 0:
        return 0.0
    return abs(fill_px) * abs(qty) * rate


def run_backtest(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    c = ExecConfig(**(cfg or {}))
    o = _col(df, "open", c.use_adjusted)
    l = _col(df, "low", c.use_adjusted)
    cl = _col(df, "close", c.use_adjusted)
    x = df[[o, l, cl]].dropna().copy()
    x = x[(x[[o, l, cl]] > 0).all(axis=1)]
    sma_f = x[cl].rolling(c.n_fast, min_periods=c.n_fast).mean()
    sma_s = x[cl].rolling(c.n_slow, min_periods=c.n_slow).mean()
    diff_prev = sma_f.shift(1) - sma_s.shift(1)
    signal_next = (diff_prev > c.epsilon).astype("Int64")
    prev_low = x[l].rolling(c.donchian_n, min_periods=c.donchian_n).min().shift(1)
    idx = x.index
    equity = c.initial_equity
    cash = c.initial_equity
    pos = 0.0
    avg_entry = 0.0
    fees_cum = 0.0
    slip_cum = 0.0
    logs: List[Dict[str, Any]] = []
    e_curve: List[Tuple[pd.Timestamp, float, float]] = []
    pending: Optional[Dict[str, Any]] = None
    for t in idx:
        if pending is not None:
            side = pending["side"]
            reason = pending["reason"]
            if side == "sell" and pos > 0:
                px_open = float(x.at[t, o])
                px_fill = _fill_price(px_open, c.slippage, "sell")
                qty = pos
                slip_amt = px_open * c.slippage * qty
                fee = _commission_cost(px_fill, qty, c.commission)
                pnl_realized = (px_fill - avg_entry) * qty
                cash += px_fill * qty - fee
                equity = cash
                fees_cum += fee
                slip_cum += slip_amt
                logs.append({
                    "t_ordered": t.isoformat(),
                    "side": "sell",
                    "reason": reason,
                    "qty": float(qty),
                    "px_open": float(px_open),
                    "px_fill": float(px_fill),
                    "fee": float(fee),
                    "slip": float(slip_amt),
                    "pnl_realized": float(pnl_realized),
                    "E_after": float(equity),
                })
                pos = 0.0
                avg_entry = 0.0
                pending = None
            elif side == "buy" and pos == 0:
                px_open = float(x.at[t, o])
                stop_prev = float(prev_low.get(t, float("nan")))
                R = c.f * equity
                qty = _risk_qty_price(R, px_open, stop_prev, c.lot_step)
                if qty > 0:
                    px_fill = _fill_price(px_open, c.slippage, "buy")
                    slip_amt = px_open * c.slippage * qty
                    fee = _commission_cost(px_fill, qty, c.commission)
                    cash -= px_fill * qty + fee
                    fees_cum += fee
                    slip_cum += slip_amt
                    pos = qty
                    avg_entry = px_fill
                    logs.append({
                        "t_ordered": t.isoformat(),
                        "side": "buy",
                        "reason": reason,
                        "qty": float(qty),
                        "px_open": float(px_open),
                        "px_fill": float(px_fill),
                        "fee": float(fee),
                        "slip": float(slip_amt),
                        "pnl_realized": 0.0,
                        "E_after": float(equity),
                    })
                pending = None
        e_curve.append((t, float(equity), float(pos)))
        stop_hit = False
        if pos > 0:
            pl = prev_low.get(t, float("nan"))
            lt = float(x.at[t, l])
            if pd.notna(pl) and lt <= float(pl):
                stop_hit = True
        sig = signal_next.get(t, pd.NA)
        want_long = (sig == 1)
        order_for_next: Optional[Dict[str, Any]] = None
        if pos > 0 and stop_hit:
            order_for_next = {"side": "sell", "reason": "stop"}
        elif pos > 0 and not want_long:
            order_for_next = {"side": "sell", "reason": "signal"}
        elif pos == 0 and want_long:
            order_for_next = {"side": "buy", "reason": "signal"}
        pending = order_for_next
    e_df = pd.DataFrame(e_curve, columns=["t", "E", "position"]).set_index("t")
    e_df["E_max"] = e_df["E"].cummax()
    e_df["drawdown"] = e_df["E"] / e_df["E_max"] - 1.0
    logs_df = pd.DataFrame(logs)
    wins = int(logs_df.loc[logs_df["side"] == "sell", "pnl_realized"].gt(0).sum()) if not logs_df.empty else 0
    losses = int(logs_df.loc[logs_df["side"] == "sell", "pnl_realized"].lt(0).sum()) if not logs_df.empty else 0
    pnl_total = float(logs_df.loc[logs_df["side"] == "sell", "pnl_realized"].sum()) if not logs_df.empty else 0.0
    ret_total = (equity / c.initial_equity) - 1.0
    metrics = {
        "trades": int((logs_df["side"] == "sell").sum()) if not logs_df.empty else 0,
        "wins": wins,
        "losses": losses,
        "pnl_total": pnl_total,
        "fees_cum": float(fees_cum),
        "slip_cum": float(slip_cum),
        "E0": float(c.initial_equity),
        "E_final": float(equity),
        "return_total": float(ret_total),
        "mdd": float(e_df["drawdown"].min()) if not e_df.empty else 0.0,
    }
    meta = {
        "config": {
            "n_fast": c.n_fast,
            "n_slow": c.n_slow,
            "epsilon": c.epsilon,
            "donchian_n": c.donchian_n,
            "f": c.f,
            "lot_step": c.lot_step,
            "commission": c.commission,
            "slippage": c.slippage,
            "asset_type": c.asset_type,
            "point_value": c.point_value,
            "initial_equity": c.initial_equity,
            "use_adjusted": c.use_adjusted,
            "name": c.name,
            "h_max": c.h_max,
        },
        "columns_used": {"open": o, "high": _col(df, "high", c.use_adjusted), "low": l, "close": cl},
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    return {"logs": logs_df, "equity_curve": e_df, "metrics": metrics, "meta": meta}


__all__ = ["ExecConfig", "run_backtest"]

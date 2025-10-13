# simulation/engine.py
"""
Backtest Engine (multi-asset, portfolio)

의도: 사전에 계산된 전략 계획(신호, 스탑, 사이징, 리밸런싱)을 입력받아,
      정해진 규칙(Stop > Signal > Rebalance)에 따라 포트폴리오 시뮬레이션을 실행한다.
      전략 계산의 책임은 오케스트레이터(backtest.py)에 있다.

출력: trades(DataFrame), equity_curve(DataFrame), metrics(dict), run_meta(dict).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .execution import open_eff


def _mdd(equity: pd.Series) -> float:
    """자본 곡선 Series로부터 최대 자본 하락(Maximum Drawdown)을 계산한다."""
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _metrics(equity: pd.Series, initial_equity: float) -> Dict[str, Any]:
    """자본 곡선으로부터 최종 성과 지표를 계산한다."""
    last = float(equity.iloc[-1]) if len(equity) else float(initial_equity)
    total_return = (last / float(initial_equity)) - 1.0 if initial_equity > 0 else 0.0
    return {
        "initial_equity": float(initial_equity),
        "final_equity": last,
        "total_return": float(total_return),
        "mdd": _mdd(equity),
    }


def _run_meta(
    *,
    prices: pd.DataFrame,
    initial_equity: float,
    lot_step: float,
    commission_rate: float,
    slip: float,
    price_step: float,
    base_currency: Optional[str],
    snapshot_meta: Optional[Dict[str, Any]],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """실행 메타데이터를 생성한다."""
    used_cols = {
        col[1] for col in prices.columns if isinstance(col, tuple) and len(col) > 1
    }
    price_cols = {
        "open": "open_adj" if "open_adj" in used_cols else "open",
        "close": "close_adj" if "close_adj" in used_cols else "close",
    }

    rm: Dict[str, Any] = {
        "engine_mode": "portfolio-plan-driven",
        "initial_equity": float(initial_equity),
        "price_columns_used": price_cols,
        "base_currency": base_currency,
        "cash_flow_source": None,
        "target_weights_source": None,
        "instrument_registry_hash": (snapshot_meta or {}).get(
            "instrument_registry_hash"
        ),
        "params": {
            "lot_step": float(lot_step),
            "price_step": float(price_step),
            "commission_rate": float(commission_rate),
            "slip": float(slip),
            **params,
        },
    }
    if snapshot_meta:
        rm["snapshot"] = snapshot_meta
    return rm


# 엔진

@dataclass
class Trade:
    """단일 거래의 모든 정보를 기록하는 데이터 클래스."""
    ts: pd.Timestamp
    symbol: str
    side: str
    reason: str
    qty: float
    price: float
    commission: float
    realized_pnl: float
    equity_after: float
    cash_after: float
    pos_after: float


def run(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    stops: pd.DataFrame,
    sizing_spec: Dict[str, pd.DataFrame],
    rebalancing_spec: Dict[str, pd.DataFrame],
    *,
    lot_step: float,
    commission_rate: float,
    slip: float,
    initial_equity: float,
    price_step: float = 0.0,
    base_currency: Optional[str] = None,
    snapshot_meta: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """포트폴리오 백테스트 엔진. Stop > Signal > Rebalance 우선순위 적용."""
    params = params or {}
    if prices.empty:
        equity_curve = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"), columns=["equity"]).fillna(0.0)
        return {
            "trades": pd.DataFrame(),
            "equity_curve": equity_curve,
            "metrics": _metrics(equity_curve["equity"], initial_equity),
            "run_meta": _run_meta(
                prices=prices,
                initial_equity=initial_equity,
                lot_step=lot_step,
                commission_rate=commission_rate,
                slip=slip,
                price_step=price_step,
                base_currency=base_currency,
                snapshot_meta=snapshot_meta,
                params=params,
            ),
        }

    symbols = list(prices.columns.get_level_values(0).unique())

    # 상태 변수 초기화
    cash: float = float(initial_equity)
    positions: Dict[str, float] = {s: 0.0 for s in symbols}
    avg_entries: Dict[str, float] = {s: 0.0 for s in symbols}

    trades: List[Trade] = []
    equity_series: List[float] = []

    idx = prices.index
    for i, ts in enumerate(idx):
        # 첫 날은 초기 자본만 기록
        if i == 0:
            equity_series.append(initial_equity)
            continue

        ts_prev = idx[i - 1]
        orders_today = []

        # 1. 스탑(Stop) 주문 생성
        for sym in symbols:
            if positions[sym] > 0 and stops.loc[ts_prev, sym]:
                orders_today.append({"symbol": sym, "side": "sell", "qty": positions[sym], "reason": "stop"})

        # 2. 신호(Signal) 주문 생성
        stop_symbols_today = {o["symbol"] for o in orders_today if o["reason"] == "stop"}
        for sym in symbols:
            if sym not in stop_symbols_today and positions[sym] == 0 and signals.loc[ts, sym] == 1:
                spec_row = sizing_spec[sym].loc[ts]
                f, stop_level = spec_row["f"], spec_row["stop_level"]
                entry_px_est = prices.loc[ts_prev, (sym, "close_adj")]
                D = entry_px_est - stop_level
                if D > 0:
                    current_total_equity = cash + sum(positions[s] * prices.loc[ts_prev, (s, "close_adj")] for s in symbols if pd.notna(prices.loc[ts_prev, (s, "close_adj")]))
                    qty = np.floor((f * current_total_equity) / D / lot_step) * lot_step
                    if qty > 0:
                         orders_today.append({"symbol": sym, "side": "buy", "qty": qty, "reason": "signal"})

        # 3. 리밸런싱(Rebalancing) 주문 생성
        buy_notional = rebalancing_spec["buy_notional"].loc[ts]
        sell_notional = rebalancing_spec["sell_notional"].loc[ts]
        for sym, notional in buy_notional.items():
            if notional > 0:
                 px_est = prices.loc[ts, (sym, "open_adj")]
                 if px_est > 0:
                    qty = np.floor(notional / px_est / lot_step) * lot_step
                    if qty > 0:
                        orders_today.append({"symbol": sym, "side": "buy", "qty": qty, "reason": "rebalance"})
        for sym, notional in sell_notional.items():
            if notional > 0 and positions[sym] > 0:
                px_est = prices.loc[ts, (sym, "open_adj")]
                if px_est > 0:
                    qty_to_sell = np.floor(notional / px_est / lot_step) * lot_step
                    qty = min(positions[sym], qty_to_sell)
                    if qty > 0:
                        orders_today.append({"symbol": sym, "side": "sell", "qty": qty, "reason": "rebalance"})

        # 4. 주문 상쇄(Netting) 및 우선순위 적용
        final_orders = []
        order_map: Dict[str, Dict[str, Any]] = {}
        for o in orders_today:
            sym = o["symbol"]
            side_val = 1 if o["side"] == "buy" else -1
            if sym not in order_map:
                order_map[sym] = {"qty": 0, "reason": "rebalance", "priority": 2}
            
            prio = {"stop": 0, "signal": 1, "rebalance": 2}[o["reason"]]
            if prio < order_map[sym]["priority"]:
                order_map[sym]["reason"] = o["reason"]
                order_map[sym]["priority"] = prio
            
            order_map[sym]["qty"] += o["qty"] * side_val

        for sym, data in order_map.items():
            net_qty = data["qty"]
            if net_qty != 0:
                final_orders.append({
                    "symbol": sym, 
                    "side": "buy" if net_qty > 0 else "sell",
                    "qty": abs(net_qty),
                    "reason": data["reason"],
                    "priority": data["priority"]
                })

        # 5. 최종 주문 실행
        for order in sorted(final_orders, key=lambda x: x["priority"]):
            sym, side, qty, reason = order["symbol"], order["side"], order["qty"], order["reason"]
            price_row = pd.Series(prices.loc[ts, sym].to_dict())
            exec_price = open_eff(price_row, slip=slip, side=side, price_step=price_step)
            
            if exec_price and exec_price > 0:
                commission = exec_price * qty * commission_rate
                if side == "buy":
                    if cash >= (exec_price * qty + commission):
                        cash -= (exec_price * qty + commission)
                        total_cost_prev = avg_entries[sym] * positions[sym]
                        positions[sym] += qty
                        avg_entries[sym] = (total_cost_prev + exec_price * qty) / positions[sym]
                        
                        pos_value_after = sum(p*prices.loc[ts, (s, 'close_adj')] for s, p in positions.items() if pd.notna(prices.loc[ts, (s, 'close_adj')]))
                        equity_after = cash + pos_value_after
                        trades.append(Trade(ts, sym, side, reason, qty, exec_price, commission, 0.0, equity_after, cash, positions[sym]))
                elif side == "sell":
                    if positions[sym] >= qty:
                        realized = (exec_price - avg_entries[sym]) * qty - commission
                        cash += exec_price * qty - commission
                        positions[sym] -= qty
                        if positions[sym] == 0: 
                            avg_entries[sym] = 0.0
                        
                        pos_value_after = sum(p*prices.loc[ts, (s, 'close_adj')] for s, p in positions.items() if pd.notna(prices.loc[ts, (s, 'close_adj')]))
                        equity_after = cash + pos_value_after
                        trades.append(Trade(ts, sym, side, reason, qty, exec_price, commission, realized, equity_after, cash, positions[sym]))

        # 6. 일일 자산 가치 스냅샷
        pos_value = sum(positions[sym] * prices.loc[ts, (sym, "close_adj")] for sym in symbols if pd.notna(prices.loc[ts, (sym, "close_adj")]))
        equity_today = cash + pos_value
        equity_series.append(equity_today)

    equity_curve = pd.DataFrame({"equity": equity_series}, index=idx)
    metrics = _metrics(equity_curve["equity"], initial_equity)

    trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame(
        columns=[
            "ts",
            "symbol",
            "side",
            "reason",
            "qty",
            "price",
            "commission",
            "realized_pnl",
            "equity_after",
            "cash_after",
            "pos_after",
        ]
    )

    run_meta = _run_meta(
        prices=prices,
        initial_equity=initial_equity,
        lot_step=lot_step,
        commission_rate=commission_rate,
        slip=slip,
        price_step=price_step,
        base_currency=base_currency,
        snapshot_meta=snapshot_meta,
        params=params,
    )

    return {
        "trades": trades_df,
        "equity_curve": equity_curve,
        "metrics": metrics,
        "run_meta": run_meta,
    }
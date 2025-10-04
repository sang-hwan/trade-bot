# simulation/engine.py
"""
Backtest Engine (single-asset, long-only)

타임라인: On-Open에 [Stop → Signal → (Rebalance)], Stop > Signal, 룩어헤드 금지.
가격 규약: *_adj 존재 시 우선 사용(혼용 금지).
라운딩: price_step/lot_step 적용. 라운딩 불능(None) → 해당 집행 스킵(주문 없음).

출력: trades(DataFrame), equity_curve(DataFrame), metrics(dict), run_meta(dict).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .execution import open_eff  # 체결가 산출(조정가·슬리피지·틱 라운딩)

# 내부 유틸

def _col(df: pd.DataFrame, base: str) -> str:
    """조정가 우선 컬럼 선택."""
    return f"{base}_adj" if f"{base}_adj" in df.columns else base


def _series_1d(df: pd.DataFrame, col: str) -> pd.Series:
    """df[col]이 DataFrame로 반환되거나 dtype이 object인 경우에도 1-D 수치 Series 보장."""
    if col not in df.columns:
        raise KeyError(f"'{col}' not in DataFrame")
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, -1]
    return pd.to_numeric(s, errors="coerce")


def _mdd(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _metrics(equity: pd.Series, initial_equity: float) -> Dict[str, Any]:
    last = float(equity.iloc[-1]) if len(equity) else float(initial_equity)
    total_return = (last / float(initial_equity)) - 1.0 if initial_equity else np.nan
    return {
        "initial_equity": float(initial_equity),
        "final_equity": last,
        "total_return": float(total_return),
        "mdd": _mdd(equity),
    }


def _run_meta(
    *,
    df: pd.DataFrame,
    initial_equity: float,
    f: float,
    N: int,
    epsilon: float,
    lot_step: float,
    commission_rate: float,
    slip: float,
    V: Optional[float],
    PV: Optional[float],
    price_step: float,
    base_currency: Optional[str],
    snapshot_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """리포트 스키마(상단 메타 포함)."""
    price_cols = {"open": _col(df, "open"), "close": _col(df, "close")}
    rm: Dict[str, Any] = {
        "engine_mode": "single-asset-long-only",
        "initial_equity": float(initial_equity),
        "price_columns_used": price_cols,
        # 상단 메타(값이 없어도 키는 존재)
        "base_currency": base_currency,
        "cash_flow_source": None,
        "target_weights_source": None,
        "instrument_registry_hash": (snapshot_meta or {}).get("instrument_registry_hash"),
        "params": {
            "f": float(f),
            "N": int(N),
            "epsilon": float(epsilon),
            "lot_step": float(lot_step),
            "price_step": float(price_step),
            "commission_rate": float(commission_rate),
            "slip": float(slip),
            "V": V,
            "PV": PV,
        },
    }
    if snapshot_meta:
        rm["snapshot"] = snapshot_meta
    return rm


# 전략 스펙(피처/신호/스탑) — 최소 구현

def _build_features(df: pd.DataFrame, sma_short: int = 10, sma_long: int = 50, N: int = 20) -> pd.DataFrame:
    """SMA(short/long), Donchian prev_low_N(t-1)/prev_low_N(t)."""
    out = df.copy()
    c = _col(df, "close")
    l = _col(df, "low")

    close_s = _series_1d(df, c)
    low_s   = _series_1d(df, l)
    
    out["sma_short"] = close_s.rolling(sma_short, min_periods=sma_short).mean()
    out["sma_long"]  = close_s.rolling(sma_long,  min_periods=sma_long ).mean()

    prev_low_tminus1 = low_s.shift(1).rolling(N, min_periods=N).min()
    prev_low_t       = low_s.rolling(N, min_periods=N).min()
    out["prev_low_N_tminus1"] = prev_low_tminus1
    out["prev_low_N"] = prev_low_t
    return out


def _build_decisions(df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """signal_next(t): Close(t-1) 결정 → Open(t) 집행."""
    out = df.copy()
    decided = (df["sma_short"] - df["sma_long"]) > float(epsilon)
    out["signal_next"] = decided.shift(1).astype("Int8")
    out["signal_next_flag"] = out["signal_next"].eq(1).fillna(False).astype(bool)
    return out


def _build_stop_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Close(t)에서 L_t ≤ prev_low_N(t-1)면 stop_hit(t)=True, Open(t+1) 집행."""
    out = df.copy()
    l = _col(df, "low")
    low_s = _series_1d(df, l)
    out["stop_hit"] = (low_s <= df["prev_low_N_tminus1"])
    out["stop_hit_flag"] = out["stop_hit"].fillna(False).astype(bool)
    return out


# 엔진

@dataclass
class Trade:
    ts: pd.Timestamp
    side: str            # "buy" | "sell"
    reason: str          # "signal" | "stop" | "rebalance"
    qty: float
    price: float
    commission: float
    slip: float
    realized_pnl: float
    equity_after: float
    position_after: float


def run(  # noqa: PLR0913 (명시적 인자 유지)
    df: pd.DataFrame,
    *,
    f: float,
    N: int,
    epsilon: float,
    lot_step: float,
    commission_rate: float,
    slip: float,
    V: Optional[float] = None,
    PV: Optional[float] = None,
    initial_equity: float = 1_000_000.0,
    price_step: float = 0.0,
    base_currency: Optional[str] = None,
    snapshot_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    단일종목 롱온리 엔진. Stop>Signal, 라운딩 불능은 '주문 없음'으로 스킵.
    """
    if df.empty:
        equity_curve = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"), columns=["equity"]).fillna(0.0)
        return {
            "trades": pd.DataFrame(),
            "equity_curve": equity_curve,
            "metrics": _metrics(equity_curve["equity"], initial_equity),
            "run_meta": _run_meta(
                df=df,
                initial_equity=initial_equity,
                f=f,
                N=N,
                epsilon=epsilon,
                lot_step=lot_step,
                commission_rate=commission_rate,
                slip=slip,
                V=V,
                PV=PV,
                price_step=price_step,
                base_currency=base_currency,
                snapshot_meta=snapshot_meta,
            ),
        }

    # 피처/신호/스탑
    sma_short, sma_long = 10, 50
    x = _build_features(df, sma_short=sma_short, sma_long=sma_long, N=N)
    x = _build_decisions(x, epsilon=epsilon)
    x = _build_stop_flags(x)

    # 상태
    cash: float = float(initial_equity)
    qty: float = 0.0
    avg_entry: float = 0.0
    pending_exit: bool = False  # 전일 stop 판정 결과

    trades: List[Trade] = []
    equity_series: List[float] = []

    close_col = _col(x, "close")

    idx = x.index
    for i, ts in enumerate(idx):
        row = x.iloc[i]

        # On-Open: 1) 스탑 체결
        skip_signal_today = False
        if pending_exit and qty > 0.0:
            exit_price = open_eff(row, slip=slip, side="sell", price_step=price_step)
            if exit_price is None:
                # 라운딩 불능 → 오늘 주문 없음(Stop > Signal 위해 신호도 스킵)
                skip_signal_today = True
            else:
                qty_executed = qty  # 실제 청산 수량 기록
                commission = abs(exit_price * qty_executed) * commission_rate
                realized = (exit_price - avg_entry) * qty_executed - commission
                cash += qty_executed * exit_price - commission
                qty = 0.0
                avg_entry = 0.0
                trades.append(
                    Trade(
                        ts=ts,
                        side="sell",
                        reason="stop",
                        qty=float(qty_executed),
                        price=float(exit_price),
                        commission=float(commission),
                        slip=float(slip),
                        realized_pnl=float(realized),
                        equity_after=float(cash),
                        position_after=float(0.0),
                    )
                )
            pending_exit = (qty > 0.0) and (exit_price is None)

        # On-Open: 2) 신호 체결
        if (not skip_signal_today) and (qty == 0.0) and bool(row["signal_next_flag"]):
            entry_px = open_eff(row, slip=slip, side="buy", price_step=price_step)
            if entry_px is not None:
                E_open = cash  # 포지션 없음 가정
                stop_level = float(row.get("prev_low_N_tminus1", np.nan))
                D = float(entry_px) - stop_level if np.isfinite(stop_level) else np.nan
                if np.isfinite(D) and D > 0.0 and f > 0.0 and lot_step > 0.0:
                    Q = np.floor((f * E_open) / D)
                    Q_exec = np.floor(Q / lot_step) * lot_step
                    if Q_exec > 0:
                        notional = Q_exec * entry_px
                        commission = abs(notional) * commission_rate
                        if cash >= (notional + commission):
                            cash -= notional + commission
                            qty += Q_exec
                            avg_entry = entry_px
                            trades.append(
                                Trade(
                                    ts=ts,
                                    side="buy",
                                    reason="signal",
                                    qty=float(Q_exec),
                                    price=float(entry_px),
                                    commission=float(commission),
                                    slip=float(slip),
                                    realized_pnl=float(0.0),
                                    equity_after=float(cash + qty * entry_px),
                                    position_after=float(qty),
                                )
                            )
                # D<=0, 라운딩 불능, 현금부족 시 주문 없음

        # On-Close: 스탑 판정 예약(다음날 Open에 집행)
        low_col = _col(x, "low")
        low_t = float(row[low_col])
        prev_low = float(row.get("prev_low_N_tminus1", np.nan))
        if qty > 0.0 and np.isfinite(prev_low) and (low_t <= prev_low):
            pending_exit = True

        # Equity 스냅샷(*_adj 우선)
        close_px = float(row[close_col])
        equity_today = cash + qty * close_px
        equity_series.append(equity_today)

    equity_curve = pd.DataFrame({"equity": equity_series}, index=idx)
    metrics = _metrics(equity_curve["equity"], initial_equity)

    trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame(
        columns=[
            "ts",
            "side",
            "reason",
            "qty",
            "price",
            "commission",
            "slip",
            "realized_pnl",
            "equity_after",
            "position_after",
        ]
    )

    run_meta = _run_meta(
        df=df,
        initial_equity=initial_equity,
        f=f,
        N=N,
        epsilon=epsilon,
        lot_step=lot_step,
        commission_rate=commission_rate,
        slip=slip,
        V=V,
        PV=PV,
        price_step=price_step,
        base_currency=base_currency,
        snapshot_meta=snapshot_meta,
    )

    return {
        "trades": trades_df,
        "equity_curve": equity_curve,
        "metrics": metrics,
        "run_meta": run_meta,
    }

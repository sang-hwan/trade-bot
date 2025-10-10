# simulation/outputs.py
"""
산출물 유틸(표준 스키마 고정).
- trades / equity_curve / metrics / run_meta 생성
- MDD = min(equity / cummax(equity) - 1)
- total_return = last_equity / initial_equity - 1
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import is_dataclass
from typing import Any
import math

import pandas as pd

__all__ = [
    "make_trades_df",
    "make_equity_curve",
    "compute_metrics",
    "build_run_meta",
    "finalize_outputs",
]


# ---------- 표준 DataFrame 구축 ----------

def make_trades_df(trades: Iterable[object]) -> pd.DataFrame:
    """거래 로그 시퀀스 → trades DataFrame(ts 오름차순). Dataclass/매핑 지원."""
    rows: list[dict[str, Any]] = []
    for t in trades:
        if is_dataclass(t):
            rows.append(t.__dict__)
        elif isinstance(t, Mapping):
            rows.append(dict(t))
        else:
            raise TypeError("trade item must be dataclass or mapping")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "ts" in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)
    return df


def make_equity_curve(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """스냅샷 행 시퀀스 → equity_curve (index=ts)."""
    ec = pd.DataFrame(list(rows))
    if ec.empty:
        return ec
    if "ts" not in ec.columns:
        raise ValueError("equity curve rows must include 'ts'")
    return ec.set_index("ts").sort_index()


# ---------- 메트릭 집계 ----------

def _mdd(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _trade_stats(trades_df: pd.DataFrame) -> dict[str, float]:
    if trades_df.empty or "realized_pnl" not in trades_df.columns:
        return {"trades": 0.0, "wins": 0.0, "losses": 0.0, "win_rate": 0.0, "payoff": 0.0, "profit_factor": 0.0}

    realized = trades_df["realized_pnl"].astype(float)
    mask_close = realized.ne(0.0)
    if not mask_close.any():
        return {"trades": float(len(trades_df)), "wins": 0.0, "losses": 0.0, "win_rate": 0.0, "payoff": 0.0, "profit_factor": 0.0}

    r = realized[mask_close]
    wins = r[r > 0.0]
    losses = r[r < 0.0]

    n_win = int((wins > 0.0).sum())
    n_loss = int((losses < 0.0).sum())
    win_rate = (n_win / (n_win + n_loss)) if (n_win + n_loss) > 0 else 0.0

    avg_win = float(wins.mean()) if n_win > 0 else 0.0
    avg_loss_abs = float((-losses).mean()) if n_loss > 0 else 0.0
    payoff = (avg_win / avg_loss_abs) if avg_loss_abs > 0.0 else 0.0

    sum_win = float(wins.sum())
    sum_loss_abs = float((-losses).sum()) if n_loss > 0 else 0.0
    profit_factor = (sum_win / sum_loss_abs) if sum_loss_abs > 0.0 else 0.0

    return {
        "trades": float(len(trades_df)),
        "wins": float(n_win),
        "losses": float(n_loss),
        "win_rate": float(win_rate),
        "payoff": float(payoff),
        "profit_factor": float(profit_factor),
    }


def compute_metrics(
    equity_curve: pd.DataFrame,
    *,
    initial_equity: float,
    trades_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """total_return, mdd, 거래 통계(선택) 집계."""
    if equity_curve.empty or "equity" not in equity_curve.columns:
        total_return = 0.0
        mdd = 0.0
    else:
        last = float(equity_curve["equity"].iloc[-1])

        # 초기자본 0/누락 방어(0-division 예방)
        if initial_equity is None or initial_equity <= 0.0 or math.isnan(float(initial_equity)):
            trade_stats = _trade_stats(trades_df if trades_df is not None else pd.DataFrame())
            return {
                "total_return": 0.0,
                "mdd": 0.0,
                **trade_stats,
            }

        total_return = (last / float(initial_equity)) - 1.0
        mdd = _mdd(equity_curve["equity"].astype(float))

    trade_stats = _trade_stats(trades_df if trades_df is not None else pd.DataFrame())
    return {"total_return": float(total_return), "mdd": float(mdd), **trade_stats}


# ---------- 메타/최종 패키징 ----------

def _extract_snapshot_fields(snapshot_meta: Mapping[str, Any]) -> dict[str, Any]:
    """엔진 리포트용 스냅샷 핵심 메타 발췌."""
    keys = (
        "source", "symbol", "start", "end", "interval", "timezone",
        "base_currency", "fx_source", "fx_source_ts", "calendar_id",
        "instrument_registry_hash", "snapshot_path", "snapshot_sha256",
        "collected_at", "rows", "columns",
    )
    return {k: snapshot_meta[k] for k in keys if k in snapshot_meta}


def build_run_meta(
    *,
    params: Mapping[str, Any],
    price_columns_used: Mapping[str, str],
    snapshot_meta: Mapping[str, Any] | None = None,
    engine_mode: str = "single-asset-long-only",
    has_rebalance: bool = False,
    fx_priced_in_data_stage: bool = True,
    version: str = "engine.v1",
) -> dict[str, Any]:
    """
    실행 메타:
    - params: 엔진 입력(f, N, lot_step, price_step, commission_rate, slip, V, PV, initial_equity 등)
    - price_columns_used: {"open": "open_adj|open", "close": "close_adj|close"}
    - snapshot_meta: 기준 통화/해시/캘린더/FX 시점 등 핵심 필드 포함
    - engine_mode: 'single-asset-long-only' 명시
    """
    meta: dict[str, Any] = {
        **dict(params),
        "price_columns_used": dict(price_columns_used),
        "engine_mode": str(engine_mode),
        "has_rebalance": bool(has_rebalance),
        "fx_priced_in_data_stage": bool(fx_priced_in_data_stage),
        "version": str(version),
    }
    if snapshot_meta is not None:
        meta["snapshot"] = _extract_snapshot_fields(snapshot_meta)
    return meta


def finalize_outputs(
    trades_df: pd.DataFrame,
    equity_curve: pd.DataFrame,
    metrics: Mapping[str, Any],
    run_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """최종 출력 딕셔너리."""
    return {
        "trades": trades_df,
        "equity_curve": equity_curve,
        "metrics": dict(metrics),
        "run_meta": dict(run_meta),
    }

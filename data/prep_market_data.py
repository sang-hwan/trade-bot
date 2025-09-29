from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class CostMeta:
    commission: float = 0.0
    slippage: float = 0.0
    lot_step: float = 1.0


def _to_utc(ts: pd.Series) -> pd.DatetimeIndex:
    return pd.to_datetime(ts, utc=True, errors="coerce")


def _dedup_sort_set_index(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    d = df.dropna(subset=[timestamp_col]).copy()
    d[timestamp_col] = _to_utc(d[timestamp_col])
    d = d.sort_values(timestamp_col).drop_duplicates(subset=[timestamp_col], keep="last")
    d = d.set_index(timestamp_col)
    d.index.name = "timestamp"
    return d


def _apply_calendar(
    df: pd.DataFrame, drop_weekends: bool, holidays: Optional[Sequence[object]]
) -> pd.DataFrame:
    d = df
    if drop_weekends:
        d = d[d.index.dayofweek < 5]
    if holidays:
        h = pd.DatetimeIndex(pd.to_datetime(holidays, utc=True)).normalize()
        d = d[~d.index.normalize().isin(h)]
    return d


def _quality_gate(df: pd.DataFrame) -> pd.DataFrame:
    req = ["open", "high", "low", "close"]
    d = df.dropna(subset=req)
    for c in req:
        d = d[d[c] > 0]
    mins = d[["open", "high", "close"]].min(axis=1)
    maxs = d[["open", "high", "close"]].max(axis=1)
    return d[(d["low"] <= mins) & (maxs <= d["high"])]


def _full_adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if "adj_close" not in df.columns:
        return df
    d = df.dropna(subset=["close", "adj_close"]).copy()
    a = d["adj_close"] / d["close"]
    d["open_adj"] = d["open"] * a
    d["high_adj"] = d["high"] * a
    d["low_adj"] = d["low"] * a
    d["close_adj"] = d["close"] * a
    return d


def prepare_market_data(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    drop_weekends: bool = False,
    holidays: Optional[Sequence[object]] = None,
    compute_adjusted: bool = True,
    cost_meta: Optional[CostMeta] = None,
) -> Tuple[pd.DataFrame, CostMeta, Dict[str, object]]:
    n_in = len(df)
    d = _dedup_sort_set_index(df, timestamp_col)
    d = _apply_calendar(d, drop_weekends=drop_weekends, holidays=holidays)
    d = _quality_gate(d)
    if compute_adjusted:
        d = _full_adjust_ohlc(d)
    if "open" not in d.columns:
        raise ValueError("open column required")
    cost = cost_meta or CostMeta()
    rep: Dict[str, object] = {
        "rows_in": n_in,
        "rows_out": len(d),
        "first_ts": d.index.min() if len(d) else None,
        "last_ts": d.index.max() if len(d) else None,
        "tz": "UTC",
        "has_adjusted": all(c in d.columns for c in ("open_adj", "high_adj", "low_adj", "close_adj")),
    }
    return d, cost, rep


def snapshot_parquet(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    name: str = "snapshot",
    version: Optional[str] = None,
    engine: str = "pyarrow",
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = df.reset_index().to_csv(index=False).encode("utf-8")
    digest = sha256(payload).hexdigest()
    ver = version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    fname = f"{name}_{digest[:8]}_v{ver}.parquet"
    path = out_dir / fname
    df.to_parquet(path, engine=engine)
    return {"path": str(path), "sha256": digest, "version": ver}

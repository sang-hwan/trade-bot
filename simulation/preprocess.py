from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Iterable
from pathlib import Path
from hashlib import sha256
from datetime import datetime, date

import pandas as pd


@dataclass
class PreprocessMeta:
    rows_before: int
    rows_after: int
    start: Optional[str]
    end: Optional[str]
    timezone: str
    columns: Tuple[str, ...]
    has_adjusted: bool
    snapshot_path: Optional[str]
    snapshot_sha256: Optional[str]


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_parquet_snapshot(df: pd.DataFrame, out_dir: Optional[str], name: str) -> Tuple[Optional[str], Optional[str]]:
    if not out_dir:
        return None, None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    path = out / f"{name}_{ts}.parquet"
    df.to_parquet(path, index=True)
    return str(path), _file_sha256(path)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "open": "open",
        "o": "open",
        "high": "high",
        "h": "high",
        "low": "low",
        "l": "low",
        "close": "close",
        "c": "close",
        "adj close": "adj_close",
        "adj_close": "adj_close",
        "adjusted close": "adj_close",
        "timestamp": "timestamp",
        "date": "timestamp",
        "datetime": "timestamp",
    }
    cols = [mapping.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns]
    out = df.copy()
    out.columns = cols
    return out


def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        idx = pd.to_datetime(out["timestamp"], utc=True)
        out = out.drop(columns=["timestamp"])
        out.index = idx
    elif not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("timestamp column or DatetimeIndex is required")
    else:
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
    out.index.name = "timestamp"
    return out


def _drop_calendar(df: pd.DataFrame, drop_weekends: bool, holidays: Optional[Iterable[date]]) -> pd.DataFrame:
    out = df
    if drop_weekends:
        out = out[~out.index.weekday.isin((5, 6))]
    if holidays:
        hol = pd.to_datetime(list(holidays)).tz_localize("UTC")
        out = out[~out.index.normalize().isin(hol.normalize())]
    return out


def _quality_gate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        raise ValueError(f"missing required columns: {sorted(missing)}")
    out = df.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    bounds_ok = (out["low"] <= out[["open", "close", "high"]].min(axis=1)) & (out[["open", "close", "high"]].max(axis=1) <= out["high"])
    out = out[bounds_ok]
    if out.empty:
        raise ValueError("no rows remain after quality gate")
    return out


def _compute_adjusted_ohlc(df: pd.DataFrame, compute_adjusted: bool) -> Tuple[pd.DataFrame, bool]:
    if compute_adjusted and "adj_close" in df.columns and (df["adj_close"] > 0).all():
        ratio = df["adj_close"] / df["close"]
        out = df.copy()
        out["open_adj"] = out["open"] * ratio
        out["high_adj"] = out["high"] * ratio
        out["low_adj"] = out["low"] * ratio
        out["close_adj"] = out["close"] * ratio
        return out, True
    return df.copy(), False


def preprocess_market_data(
    df: pd.DataFrame,
    *,
    drop_weekends: bool = False,
    holidays: Optional[Iterable[date]] = None,
    compute_adjusted: bool = True,
    snapshot_dir: Optional[str] = None,
    name: str = "preprocessed",
) -> Dict[str, Any]:
    rows_before = int(df.shape[0])
    out = _normalize_columns(df)
    out = _to_utc_index(out)
    out = _drop_calendar(out, drop_weekends=drop_weekends, holidays=holidays)
    out = _quality_gate_ohlc(out)
    out, has_adj = _compute_adjusted_ohlc(out, compute_adjusted=compute_adjusted)
    rows_after = int(out.shape[0])
    start = out.index.min().isoformat()
    end = out.index.max().isoformat()
    snapshot_path, snapshot_hash = _write_parquet_snapshot(out, snapshot_dir, name)
    meta = PreprocessMeta(
        rows_before=rows_before,
        rows_after=rows_after,
        start=start,
        end=end,
        timezone="UTC",
        columns=tuple(map(str, out.columns)),
        has_adjusted=has_adj,
        snapshot_path=snapshot_path,
        snapshot_sha256=snapshot_hash,
    )
    return {"df": out, "meta": meta}


__all__ = ["PreprocessMeta", "preprocess_market_data"]

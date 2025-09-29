from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from hashlib import sha256
from datetime import datetime

import pandas as pd


@dataclass
class SourceMeta:
    source: str
    symbol: Optional[str]
    path: Optional[str]
    start: Optional[str]
    end: Optional[str]
    interval: Optional[str]
    timezone: Optional[str]
    rows: int
    cols: int
    columns: Tuple[str, ...]
    collected_at: str
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


def collect_from_yahoo(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    timezone: Optional[str] = None,
    snapshot_dir: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        import yfinance as yf
    except (ModuleNotFoundError, ImportError) as e:
        raise ImportError("yfinance is required for Yahoo collection") from e
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(0, axis=1)
    df.index.name = "timestamp"
    snapshot_path, snapshot_hash = _write_parquet_snapshot(df, snapshot_dir, f"raw_{symbol}")
    meta = SourceMeta(
        source="yahoo",
        symbol=symbol,
        path=None,
        start=start,
        end=end,
        interval=interval,
        timezone=timezone,
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        columns=tuple(map(str, df.columns)),
        collected_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        snapshot_path=snapshot_path,
        snapshot_sha256=snapshot_hash,
    )
    return {"df": df, "meta": meta}


def collect_from_file(
    path: str,
    timezone: Optional[str] = None,
    snapshot_dir: Optional[str] = None,
) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    suffix = p.suffix.lower()
    if suffix in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    elif suffix in (".csv", ".txt"):
        df = pd.read_csv(p)
    elif suffix in (".feather", ".ft"):
        df = pd.read_feather(p)
    else:
        raise ValueError(f"unsupported file type: {suffix}")
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(0, axis=1)
    snapshot_path, snapshot_hash = _write_parquet_snapshot(df, snapshot_dir, p.stem)
    meta = SourceMeta(
        source="file",
        symbol=None,
        path=str(p),
        start=None,
        end=None,
        interval=None,
        timezone=timezone,
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        columns=tuple(map(str, df.columns)),
        collected_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        snapshot_path=snapshot_path,
        snapshot_sha256=snapshot_hash,
    )
    return {"df": df, "meta": meta}


def collect(
    source: str,
    *,
    symbol: Optional[str] = None,
    path: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    timezone: Optional[str] = None,
    snapshot_dir: Optional[str] = None,
) -> Dict[str, Any]:
    s = source.lower()
    if s == "yahoo":
        if not symbol:
            raise ValueError("symbol is required for yahoo")
        return collect_from_yahoo(symbol=symbol, start=start, end=end, interval=interval, timezone=timezone, snapshot_dir=snapshot_dir)
    if s == "file":
        if not path:
            raise ValueError("path is required for file")
        return collect_from_file(path=path, timezone=timezone, snapshot_dir=snapshot_dir)
    raise ValueError(f"unsupported source: {source}")


__all__ = [
    "SourceMeta",
    "collect",
    "collect_from_yahoo",
    "collect_from_file",
]

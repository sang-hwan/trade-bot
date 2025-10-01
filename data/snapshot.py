# snapshot.py
"""
스냅샷 고정(Parquet 저장 + SHA-256 해시 + 메타 생성).
- 데이터는 이미 품질 검증/조정이 끝났다고 가정(수정 없음).
- 인덱스: DatetimeIndex(tz='UTC')를 그대로 보존.

공개 API
- write(df, *, source, symbol, start, end, interval, out_dir, filename=None, timezone='UTC') -> SnapshotMeta
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone as _tz
from hashlib import sha256
from pathlib import Path

import pandas as pd

__all__ = ["SnapshotMeta", "SnapshotError", "write"]


class SnapshotError(ValueError):
    """스냅샷 단계 예외."""


@dataclass(frozen=True)
class SnapshotMeta:
    source: str
    symbol: str
    start: str | None
    end: str | None
    interval: str
    rows: int
    columns: tuple[str, ...]
    snapshot_path: str
    snapshot_sha256: str
    collected_at: str
    timezone: str


def _sanitize_filename(s: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    out = "".join(ch if ch in allowed else "-" for ch in s).strip("-")
    return out or "snapshot"


def _ensure_parquet_suffix(name: str) -> str:
    return name if name.lower().endswith(".parquet") else f"{name}.parquet"


def _now_utc_iso() -> str:
    return datetime.now(tz=_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MiB
            h.update(chunk)
    return h.hexdigest()


def _default_filename(source: str, symbol: str, interval: str, end: str | None) -> str:
    ts = (
        pd.to_datetime(end, utc=True).strftime("%Y%m%d%H%M%SZ")
        if end
        else datetime.now(tz=_tz.utc).strftime("%Y%m%d%H%M%SZ")
    )
    base = f"{_sanitize_filename(source)}_{_sanitize_filename(symbol)}_{_sanitize_filename(interval)}_{ts}"
    return f"{base}.parquet"


def write(
    df: pd.DataFrame,
    *,
    source: str,
    symbol: str,
    start: str | None,
    end: str | None,
    interval: str,
    out_dir: str | Path,
    filename: str | None = None,
    timezone: str = "UTC",
    parquet_engine: str | None = None,
    parquet_compression: str | None = None,
) -> SnapshotMeta:
    """
    Parquet 저장 → 파일 해시 산출 → SnapshotMeta 반환.
    - df.index: DatetimeIndex(tz='UTC') 필수(변환/수정 없음)
    - filename 미지정 시 '{source}_{symbol}_{interval}_{YYYYMMDDhhmmssZ}.parquet'
    - parquet_*: pandas.to_parquet에 전달
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise SnapshotError("df.index는 DatetimeIndex여야 합니다.")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise SnapshotError("df.index 타임존은 tz='UTC' 여야 합니다.")

    out_dir = Path(out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise SnapshotError(f"out_dir가 디렉터리가 아닙니다: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = filename or _default_filename(source, symbol, interval, end)
    fname = _ensure_parquet_suffix(fname)
    path = out_dir / fname

    df.to_parquet(path, engine=parquet_engine, compression=parquet_compression, index=True)
    digest = _file_sha256(path)

    return SnapshotMeta(
        source=source,
        symbol=symbol,
        start=start,
        end=end,
        interval=interval,
        rows=int(len(df)),
        columns=tuple(map(str, df.columns)),
        snapshot_path=str(path),
        snapshot_sha256=digest,
        collected_at=_now_utc_iso(),
        timezone=timezone,
    )

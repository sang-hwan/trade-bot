# snapshot.py
"""
스냅샷 고정(Parquet 저장 + SHA-256 해시 + 메타 생성).
- 전제: 입력은 품질 검증/조정 완료, 수정 없음.
- 인덱스: DatetimeIndex(tz='UTC') 보존.

공개 API
- write(df, *, source, symbol, start, end, interval, out_dir, filename=None, timezone='UTC') -> SnapshotMeta
"""

from __future__ import annotations

# 표준 라이브러리 우선
from dataclasses import dataclass
from datetime import datetime, timezone as _tz
from hashlib import sha256
from pathlib import Path

# 서드파티
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


def _to_parquet_with_fallback(
    df: pd.DataFrame,
    path: Path,
    *,
    engine_pref: str | None,
    compression: str | None,
) -> None:
    """엔진 우선순위에 따라 저장 시도: 지정값 → 'pyarrow' → 'fastparquet' → 기본탐지."""
    tried: list[tuple[str | None, str]] = []
    engines: list[str | None] = []
    if engine_pref:
        engines.append(engine_pref)
    engines.extend([e for e in ("pyarrow", "fastparquet") if e != engine_pref])
    engines.append(None)  # pandas 기본 탐지

    for eng in engines:
        try:
            df.to_parquet(path, index=True, engine=eng, compression=compression)
            return
        except Exception as e:  # 다양한 엔진 예외를 간결히 수집
            tried.append((eng, repr(e)))

    msgs = " | ".join([f"engine={eng!r}: {err}" for eng, err in tried])
    order = " -> ".join([repr(e) for e in engines])
    raise SnapshotError(f"Parquet write failed. Tried: {order} | errors: {msgs}")


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
    - df.index: DatetimeIndex(tz='UTC') 필수(변환 없음)
    - filename 미지정 시 '{source}_{symbol}_{interval}_{YYYYMMDDhhmmssZ}.parquet'
    - parquet_*는 pandas.to_parquet 인자로 전달(엔진은 폴백 적용)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise SnapshotError("df.index는 DatetimeIndex여야 합니다.")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise SnapshotError("df.index 타임존은 tz='UTC' 여야 합니다.")

    out_dir = Path(out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise SnapshotError(f"out_dir가 디렉터리가 아닙니다: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = _ensure_parquet_suffix(filename or _default_filename(source, symbol, interval, end))
    path = out_dir / fname

    _to_parquet_with_fallback(df, path, engine_pref=parquet_engine, compression=parquet_compression)

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

# snapshot.py
"""
스냅샷 고정(Parquet 저장 + SHA-256 해시 + 메타 생성).

공개 API
- write(
    df, *,
    source, symbol, start, end, interval, out_dir,
    filename=None, timezone='UTC',
    parquet_engine=None, parquet_compression=None,
    # 옵션 메타(제공 시에만 기록)
    base_currency=None, fx_source=None, fx_source_ts=None,
    calendar_id=None, instrument_registry_hash=None
  ) -> SnapshotMeta
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone as _tz
from hashlib import sha256
from pathlib import Path
import json

import pandas as pd

__all__ = ["SnapshotMeta", "SnapshotError", "write"]


class SnapshotError(ValueError):
    """스냅샷 단계 예외."""


@dataclass(frozen=True)
class SnapshotMeta:
    # 기본 메타
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
    # 옵션 메타
    base_currency: str | None = None
    fx_source: str | None = None
    fx_source_ts: str | None = None
    calendar_id: str | None = None
    instrument_registry_hash: str | None = None


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
    """지정 엔진 → 'pyarrow' → 'fastparquet' → 기본탐지 순으로 시도, 실패 시 누적 오류 포함해 SnapshotError."""
    tried: list[tuple[str | None, str]] = []
    engines: list[str | None] = []
    if engine_pref:
        engines.append(engine_pref)
    engines.extend([e for e in ("pyarrow", "fastparquet") if e != engine_pref])
    engines.append(None)

    for eng in engines:
        try:
            df.to_parquet(path, index=True, engine=eng, compression=compression)
            return
        except Exception as e:
            tried.append((eng, repr(e)))

    msgs = " | ".join([f"engine={eng!r}: {err}" for eng, err in tried])
    order = " -> ".join([repr(e) for e in engines])
    raise SnapshotError(f"Parquet write failed. Tried: {order} | errors: {msgs}")


def _norm_str(x: str | None) -> str | None:
    """공백/빈 문자열을 None으로 정규화."""
    if x is None:
        return None
    x = str(x).strip()
    return x if x else None


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
    base_currency: str | None = None,
    fx_source: str | None = None,
    fx_source_ts: str | None = None,
    calendar_id: str | None = None,
    instrument_registry_hash: str | None = None,
) -> SnapshotMeta:
    """
    Parquet 저장 → 파일 해시 산출 → SnapshotMeta 반환.
    - df.index: DatetimeIndex(tz='UTC') 필수
    - filename 미지정: '{source}_{symbol}_{interval}_{YYYYMMDDhhmmssZ}.parquet'
    - parquet_*: pandas.to_parquet 전달(엔진 폴백 적용)
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

    meta = SnapshotMeta(
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
        base_currency=_norm_str(base_currency),
        fx_source=_norm_str(fx_source),
        fx_source_ts=_norm_str(fx_source_ts),
        calendar_id=_norm_str(calendar_id),
        instrument_registry_hash=_norm_str(instrument_registry_hash),
    )

    # 사이드카 JSON 저장(메타 기록 실패는 스냅샷 성공과 분리)
    try:
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        with sidecar.open("w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, ensure_ascii=False, indent=2)
    except (OSError, TypeError, ValueError):
        # 파일 시스템/직렬화 오류 등은 무시(스냅샷 자체는 성공)
        pass

    return meta

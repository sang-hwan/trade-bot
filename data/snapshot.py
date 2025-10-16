# data/snapshot.py
"""
Snapshot Serialization: Persists a DataFrame to Parquet format,
calculates its SHA-256 hash, and generates comprehensive metadata.

Public API:
- write(...) -> SnapshotMeta

Key Conventions:
- The DataFrame index must be a UTC DatetimeIndex.
- The 'start' and 'end' fields in the metadata represent the actual
  min/max timestamps from the snapshot data, not the requested range.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["SnapshotMeta", "SnapshotError", "write"]


class SnapshotError(ValueError):
    """Custom exception for errors during the snapshot process."""


@dataclass(frozen=True)
class SnapshotMeta:
    """
    A container for all metadata associated with a data snapshot.
    This information is crucial for reproducibility and validation.
    """
    # Core metadata for reproducibility
    source: str
    symbol: str
    start: str  # Actual min timestamp in UTC ISO format (Z)
    end: str    # Actual max timestamp in UTC ISO format (Z)
    interval: str
    rows: int
    columns: int
    snapshot_path: str
    snapshot_sha256: str
    collected_at: str
    timezone: str  # Informational only, not used for timezone conversion

    # Supplemental metadata
    column_names: tuple[str, ...] | None = None
    requested_start: str | None = None
    requested_end: str | None = None
    base_currency: str | None = None
    fx_source: str | None = None
    fx_source_ts: str | None = None
    calendar_id: str | None = None
    instrument_registry_hash: str | None = None

    # Fields required for the validation gate
    asset_class: str | None = None
    price_currency: str | None = None
    tick_size: float | None = None
    lot_step: float | None = None
    trading_status: str | None = None


def _sanitize_filename(s: str) -> str:
    """Converts a string into a filesystem-safe filename."""
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    return "".join(c if c in allowed else "-" for c in s).strip("-") or "snapshot"


def _ensure_parquet_suffix(name: str) -> str:
    """Ensures the filename ends with the .parquet extension."""
    return name if name.lower().endswith(".parquet") else f"{name}.parquet"


def _now_utc_iso() -> str:
    """Returns the current time in UTC ISO-8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_sha256(path: Path) -> str:
    """Calculates the SHA-256 hash of a file."""
    h = sha256()
    with path.open("rb") as f:
        # Read in 1 MiB chunks for efficiency with large files
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_filename(source: str, symbol: str, interval: str, end_utc_iso: str | None) -> str:
    """Generates a default filename for the snapshot."""
    ts_str = end_utc_iso or _now_utc_iso()
    ts = pd.to_datetime(ts_str, utc=True).strftime("%Y%m%d%H%M%SZ")
    base = f"{_sanitize_filename(source)}_{_sanitize_filename(symbol)}_{_sanitize_filename(interval)}_{ts}"
    return f"{base}.parquet"


def _to_parquet_with_fallback(
    df: pd.DataFrame, path: Path, *, engine_pref: str | None, compression: str | None
) -> None:
    """Attempts to write a DataFrame to Parquet using a sequence of engines."""
    engines_to_try = []
    if engine_pref:
        engines_to_try.append(engine_pref)
    engines_to_try.extend([e for e in ("pyarrow", "fastparquet") if e != engine_pref])
    engines_to_try.append("auto")  # Let pandas auto-detect

    errors = []
    for engine in engines_to_try:
        try:
            df.to_parquet(path, index=True, engine=engine, compression=compression)
            return
        except Exception as e:
            errors.append(f"engine='{engine}': {e!r}")

    error_summary = " | ".join(errors)
    raise SnapshotError(f"Failed to write Parquet file. Tried engines {engines_to_try}. Errors: {error_summary}")


def _norm_str(value: Any) -> str | None:
    """Normalizes a value to a stripped string or None if empty."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


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
    # New arguments for validation gate compatibility
    asset_class: str | None = None,
    price_currency: str | None = None,
    tick_size: float | None = None,
    lot_step: float | None = None,
    trading_status: str | None = None,
) -> SnapshotMeta:
    """
    Saves a DataFrame to a Parquet file and returns a SnapshotMeta object.

    Args:
        df: The DataFrame to save, which must have a UTC DatetimeIndex.
        ... and other metadata fields.

    Returns:
        A SnapshotMeta object containing all metadata about the saved file.

    Raises:
        SnapshotError: If the DataFrame is empty or has an invalid index.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or str(df.index.tz) != "UTC":
        raise SnapshotError("DataFrame index must be a DatetimeIndex with UTC timezone.")
    if df.empty:
        raise SnapshotError("Cannot create a snapshot for an empty DataFrame.")

    # Determine actual time range from the data itself
    start_actual_iso = df.index.min().strftime("%Y-%m-%dT%H:%M:%SZ")
    end_actual_iso = df.index.max().strftime("%Y-%m-%dT%H:%M:%SZ")

    output_directory = Path(out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    file_name = _ensure_parquet_suffix(filename or _default_filename(source, symbol, interval, end_actual_iso))
    file_path = output_directory / file_name

    _to_parquet_with_fallback(df, file_path, engine_pref=parquet_engine, compression=parquet_compression)
    file_hash = _file_sha256(file_path)

    meta = SnapshotMeta(
        source=source,
        symbol=symbol,
        start=start_actual_iso,
        end=end_actual_iso,
        interval=interval,
        rows=df.shape[0],
        columns=df.shape[1],
        snapshot_path=str(file_path),
        snapshot_sha256=file_hash,
        collected_at=_now_utc_iso(),
        timezone=timezone,
        column_names=tuple(df.columns),
        requested_start=_norm_str(start),
        requested_end=_norm_str(end),
        base_currency=_norm_str(base_currency),
        fx_source=_norm_str(fx_source),
        fx_source_ts=_norm_str(fx_source_ts),
        calendar_id=_norm_str(calendar_id),
        instrument_registry_hash=_norm_str(instrument_registry_hash),
        asset_class=_norm_str(asset_class),
        price_currency=_norm_str(price_currency),
        tick_size=float(tick_size) if tick_size is not None else None,
        lot_step=float(lot_step) if lot_step is not None else None,
        trading_status=_norm_str(trading_status),
    )

    # Save metadata to a sidecar JSON file (best-effort)
    meta_path = file_path.with_suffix(file_path.suffix + ".meta.json")
    try:
        meta_path.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        # Failure to write the sidecar file should not abort the process
        pass

    return meta


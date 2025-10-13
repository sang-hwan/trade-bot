# validation/data_gate.py
"""
Data/Snapshot Integrity Gate

의도: 입력 스냅샷이 저장 규약을 준수하고(UTC, 표준 메타 포함),
같은 입력이면 같은 결과가 다시 나오도록 재현성을 보장한다.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd


class GateResult(TypedDict):
    passed: bool
    errors: List[str]
    warnings: List[str]
    evidence: Dict[str, Any]


@dataclass
class Artifacts:
    """검증에 필요한 산출물 핸들."""
    snapshot_parquet_path: str
    snapshot_meta: Dict[str, Any]


# 필수 스냅샷 메타 키/타입
_REQUIRED_SNAPSHOT_FIELDS: Dict[str, tuple[type, ...]] = {
    "source": (str,), "symbol": (str,), "start": (str,), "end": (str,),
    "interval": (str,), "rows": (int,), "columns": (int,),
    "snapshot_path": (str,), "snapshot_sha256": (str,), "collected_at": (str,),
    "timezone": (str,), "base_currency": (str,), "fx_source": (str, type(None)),
    "fx_source_ts": (str, type(None)), "calendar_id": (str,),
    "instrument_registry_hash": (str, type(None)),
}


def _sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    """파일의 SHA-256 해시를 계산한다."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_iso8601_utc(s: Optional[str]) -> bool:
    """문자열이 'Z'로 끝나는 UTC ISO-8601 형식인지 확인한다."""
    if not s:
        return True
    try:
        if not s.endswith("Z"):
            return False
        pd.to_datetime(s, utc=True)
        return True
    except (ValueError, TypeError):
        return False


def run(artifacts: Artifacts) -> GateResult:
    """데이터/스냅샷 무결성 게이트의 단일 진입점."""
    errors: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {}

    # (1/5) snapshot_meta 형식 검증
    meta = artifacts.snapshot_meta or {}
    for key, types in _REQUIRED_SNAPSHOT_FIELDS.items():
        if key not in meta:
            errors.append(f"[meta:missing] '{key}' 누락")
        elif not isinstance(meta.get(key), types):
            errors.append(f"[meta:type] '{key}' 타입 불일치: {type(meta.get(key)).__name__} != {types}")

    for k in ("start", "end", "collected_at", "fx_source_ts"):
        if k in meta and not _is_iso8601_utc(meta.get(k)):
            errors.append(f"[meta:iso8601] '{k}'는 UTC(Z) ISO-8601 형식이여야 합니다: {meta.get(k)!r}")

    # (2/5) Parquet 파일 로드 및 인덱스 검증
    pq_path = artifacts.snapshot_parquet_path
    if not os.path.isfile(pq_path):
        errors.append(f"[parquet:missing] 파일 없음: {pq_path}")
        return GateResult(passed=len(errors) == 0, errors=errors, warnings=warnings, evidence=evidence)

    try:
        df = pd.read_parquet(pq_path)
    except Exception as e:
        errors.append(f"[parquet:read] 로드 실패: {pq_path} ({e})")
        return GateResult(passed=len(errors) == 0, errors=errors, warnings=warnings, evidence=evidence)

    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("[index:type] 인덱스가 DatetimeIndex가 아닙니다.")
    elif df.index.tz is None or str(df.index.tz) != "UTC":
        errors.append("[index:tz] 인덱스 타임존이 UTC가 아닙니다.")
    else:
        if not df.index.is_monotonic_increasing:
            errors.append("[index:order] 인덱스가 오름차순이 아닙니다.")
        if not df.index.is_unique:
            errors.append("[index:dup] 인덱스에 중복된 값이 존재합니다.")

    # (3/5) 메타데이터와 실제 데이터 내용 대조
    if not df.empty:
        if meta.get("rows") != df.shape[0]:
            errors.append(f"[meta:rows] 메타({meta.get('rows')})와 실제 행 수({df.shape[0]})가 다릅니다.")
        if meta.get("columns") != df.shape[1]:
            errors.append(f"[meta:columns] 메타({meta.get('columns')})와 실제 열 수({df.shape[1]})가 다릅니다.")

        actual_start_str = pd.to_datetime(df.index.min(), utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
        actual_end_str = pd.to_datetime(df.index.max(), utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
        if meta.get("start") != actual_start_str:
            errors.append(f"[meta:start] 메타({meta.get('start')})와 실제 시작 시간({actual_start_str})이 다릅니다.")
        if meta.get("end") != actual_end_str:
            errors.append(f"[meta:end] 메타({meta.get('end')})와 실제 종료 시간({actual_end_str})이 다릅니다.")

    # (4/5) 파일 경로 및 해시 일치 검증
    try:
        calculated_sha = _sha256_of_file(pq_path)
        meta_sha = str(meta.get("snapshot_sha256", ""))
        if not meta_sha:
            errors.append("[meta:sha256] snapshot_sha256가 누락되었습니다.")
        elif calculated_sha.lower() != meta_sha.lower():
            errors.append(f"[meta:sha256] 해시 불일치: 계산된 값={calculated_sha}, 메타 값={meta_sha}")
    except OSError as e:
        errors.append(f"[meta:sha256-calc] 해시 계산 실패: {e}")

    meta_path = str(meta.get("snapshot_path", "")).strip()
    if not meta_path:
        errors.append("[meta:snapshot_path] snapshot_path가 누락되었습니다.")
    elif os.path.basename(meta_path) != os.path.basename(pq_path):
        errors.append(f"[meta:snapshot_path] 파일명 불일치: 메타 경로={meta_path}, 실제 경로={pq_path}")

    # (5/5) 증거 데이터 기록
    evidence.update({
        "meta_fields_checked": list(_REQUIRED_SNAPSHOT_FIELDS.keys()),
        "parquet_path_actual": pq_path,
        "sha256_calculated": locals().get("calculated_sha"),
    })

    return GateResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
    )

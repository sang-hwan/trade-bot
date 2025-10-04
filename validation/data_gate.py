# validation/data_gate.py
"""
Data/Snapshot Integrity Gate

의도: 입력 스냅샷이 저장 규약을 준수하고(UTC, 표준 메타 포함),
같은 입력이면 같은 결과가 다시 나오도록 재현성을 보장한다.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd


class GateResult(TypedDict):
    passed: bool
    errors: List[str]
    warnings: List[str]
    evidence: Dict[str, Any]


@dataclass
class Artifacts:
    snapshot_parquet_path: str
    snapshot_meta: Dict[str, Any]
    # 문서 정적 규약 검사 대상 디렉터리(README, 설계 문서 등)
    docs_dir: Optional[str] = None


# 필수 스냅샷 메타 키/타입
_REQUIRED_SNAPSHOT_FIELDS: Dict[str, tuple[type, ...]] = {
    "source": (str,),
    "symbol": (str,),
    "start": (str,),
    "end": (str,),
    "interval": (str,),
    "rows": (int,),
    "columns": (int,),
    "snapshot_path": (str,),
    "snapshot_sha256": (str,),
    "collected_at": (str,),
    "timezone": (str,),
    "base_currency": (str,),
    "fx_source": (str, type(None)),
    "fx_source_ts": (str, type(None)),
    "calendar_id": (str,),
    "instrument_registry_hash": (str, type(None)),
}

# \[...\] 금지 탐지용
_BRACKET_PATTERN = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)


def _sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_iso8601(s: Optional[str]) -> bool:
    if s is None:
        return True
    try:
        datetime.fromisoformat(s.replace("Z", "+00:00"))
        return True
    except (ValueError, TypeError):
        return False


def _to_iso_utc(ts: pd.Timestamp) -> str:
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _lint_display_math_blocks(docs_dir: Optional[str]) -> List[str]:
    """디스플레이 수식 규약 검사: $$ 블록 앞뒤 빈 줄 1줄, \[...\] 금지."""
    issues: List[str] = []
    if not docs_dir or not os.path.isdir(docs_dir):
        return issues

    md_files: List[str] = []
    for root, _, files in os.walk(docs_dir):
        for fn in files:
            if fn.lower().endswith((".md", ".mdx", ".markdown", ".txt")):
                md_files.append(os.path.join(root, fn))

    for path in md_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except OSError as e:
            issues.append(f"[doc:read-fail] {path}: {e}")
            continue

        content = "\n".join(lines)
        if _BRACKET_PATTERN.search(content):
            issues.append(f"[doc:display-math] '\\[...\\]' 사용 금지: {path}")

        in_block = False
        start_idx = -1
        for i, line in enumerate(lines):
            if line.strip() == "$$":
                if not in_block:
                    in_block = True
                    start_idx = i
                    if i == 0 or lines[i - 1].strip() != "":
                        issues.append(f"[doc:display-math] 블록 시작 전 빈 줄 필요: {path}:{i+1}")
                else:
                    in_block = False
                    end_idx = i
                    if end_idx == len(lines) - 1 or lines[end_idx + 1].strip() != "":
                        issues.append(f"[doc:display-math] 블록 종료 후 빈 줄 필요: {path}:{end_idx+1}")

        if in_block:
            issues.append(f"[doc:display-math] 닫히지 않은 $$ 블록: {path}:{start_idx+1}")

    return issues


def run(artifacts: Artifacts) -> GateResult:
    """데이터/스냅샷 무결성 게이트의 단일 진입점."""
    errors: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {}

    # 1) snapshot_meta 필수 키·타입·형식
    meta = artifacts.snapshot_meta or {}
    for key, types in _REQUIRED_SNAPSHOT_FIELDS.items():
        if key not in meta:
            errors.append(f"[meta:missing] '{key}' 누락")
            continue
        if not isinstance(meta[key], types):
            errors.append(f"[meta:type] '{key}' 타입 불일치: {type(meta[key]).__name__} != {types}")

    for k in ("start", "end", "collected_at", "fx_source_ts"):
        if k in meta and not _is_iso8601(meta.get(k)):
            errors.append(f"[meta:iso8601] '{k}' 형식 오류: {meta.get(k)!r}")

    # 2) Parquet 로드 & 인덱스 검증
    pq_path = artifacts.snapshot_parquet_path
    if not os.path.isfile(pq_path):
        return GateResult(passed=False, errors=[f"[parquet:missing] 파일 없음: {pq_path}"], warnings=[], evidence={})

    try:
        df = pd.read_parquet(pq_path)
    except Exception as e:  # pandas/엔진별 예외가 다양하여 포괄 처리
        return GateResult(passed=False, errors=[f"[parquet:read] 로드 실패: {pq_path} ({e})"], warnings=[], evidence={})

    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("[index:type] DatetimeIndex 아님")
        idx_utc = None
    else:
        if df.index.tz is None:
            errors.append("[index:tz] 타임존 정보 없음 — 반드시 UTC여야 함")
            idx_utc = df.index.tz_localize("UTC")
        else:
            try:
                idx_utc = df.index.tz_convert("UTC")
            except (TypeError, ValueError) as e:
                errors.append(f"[index:tz-convert] UTC 변환 실패: {e}")
                idx_utc = df.index

        if idx_utc is not None:
            if not idx_utc.is_monotonic_increasing:
                errors.append("[index:order] 인덱스가 오름차순이 아님")
            if not idx_utc.is_unique:
                errors.append("[index:dup] 인덱스에 중복이 존재")
            evidence["index_min_utc"] = _to_iso_utc(idx_utc.min())
            evidence["index_max_utc"] = _to_iso_utc(idx_utc.max())

    # 3) rows/columns, start/end 합치
    evidence["rows"] = int(df.shape[0])
    evidence["columns"] = int(df.shape[1])

    mr = meta.get("rows")
    mc = meta.get("columns")
    if isinstance(mr, int) and mr != evidence["rows"]:
        errors.append(f"[meta:rows] 예상({mr}) != 실제({evidence['rows']})")
    if isinstance(mc, int) and mc != evidence["columns"]:
        errors.append(f"[meta:columns] 예상({mc}) != 실제({evidence['columns']})")

    try:
        base_idx = df.index.tz_convert("UTC") if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None else df.index.tz_localize("UTC")
        min_iso = _to_iso_utc(base_idx.min())
        max_iso = _to_iso_utc(base_idx.max())
        evidence["index_min_iso"] = min_iso
        evidence["index_max_iso"] = max_iso

        mstart = str(meta.get("start", "")).replace("Z", "+00:00")
        mend = str(meta.get("end", "")).replace("Z", "+00:00")
        ms_iso = datetime.fromisoformat(mstart).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if mstart else ""
        me_iso = datetime.fromisoformat(mend).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if mend else ""
        if ms_iso and ms_iso != min_iso:
            errors.append(f"[meta:start] 예상({ms_iso}) != 실제({min_iso})")
        if me_iso and me_iso != max_iso:
            errors.append(f"[meta:end] 예상({me_iso}) != 실제({max_iso})")
    except (ValueError, TypeError, AttributeError) as e:
        errors.append(f"[meta:start-end-compare] 비교 실패: {e}")

    # 4) 파일 경로·해시 일치
    try:
        calc_sha = _sha256_of_file(pq_path)
        evidence["sha256_calc"] = calc_sha
        meta_sha = str(meta.get("snapshot_sha256", ""))
        if not meta_sha:
            errors.append("[meta:sha256] snapshot_sha256 누락")
        elif calc_sha.lower() != meta_sha.lower():
            errors.append(f"[meta:sha256] 불일치: calc={calc_sha} meta={meta_sha}")
    except OSError as e:
        errors.append(f"[meta:sha256-calc] 실패: {e}")

    try:
        meta_path = str(meta.get("snapshot_path", "")).strip()
        if meta_path:
            if os.path.basename(meta_path) != os.path.basename(pq_path):
                errors.append(f"[meta:snapshot_path] 파일명 불일치: meta={meta_path} actual={pq_path}")
        else:
            errors.append("[meta:snapshot_path] 누락")
    except (TypeError, AttributeError) as e:
        errors.append(f"[meta:snapshot_path-check] 실패: {e}")

    if not meta.get("timezone"):
        errors.append("[meta:timezone] 누락")

    # 5) 디스플레이 수식 규약
    math_issues = _lint_display_math_blocks(artifacts.docs_dir)
    if math_issues:
        errors.extend(math_issues)

    return GateResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
    )

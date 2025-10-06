# data/quality_gate.py
"""
정합성(Integrity) 검증 — 공개 API
- validate(df, *, require_volume=False, min_rows=1) -> None
- validate_meta(meta: dict, *, require_fx=True, require_calendar=True, require_steps=True) -> None
"""

from __future__ import annotations

# ── 표준 라이브러리 우선
import re
from collections.abc import Iterable
from typing import Any, Mapping

# ── 서드파티
import pandas as pd

__all__ = ["QualityError", "validate", "validate_meta"]


class QualityError(ValueError):
    """정합성 위반 예외."""


# ----------------------------- DataFrame 게이트 -----------------------------


def _assert_columns(df: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise QualityError(f"필수 컬럼 누락: {missing} (필수: {list(required)})")


def _assert_index_is_utc_dtindex(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise QualityError("인덱스가 DatetimeIndex가 아닙니다.")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise QualityError("인덱스 타임존은 반드시 UTC 여야 합니다 (tz='UTC').")


def _assert_monotonic_strict_and_no_duplicates(df: pd.DataFrame) -> None:
    if df.index.has_duplicates:
        dup_count = len(df.index) - len(df.index.drop_duplicates())
        raise QualityError(f"인덱스 중복 발견: {dup_count}개.")
    if not df.index.is_monotonic_increasing:
        raise QualityError("인덱스가 오름차순이 아닙니다(단조 증가 위반).")
    diffs = df.index.to_series().diff().dropna()
    if not (diffs > pd.Timedelta(0)).all():
        raise QualityError("인덱스에 음수 또는 0 간격이 존재합니다.")


def _as_1d_series(df: pd.DataFrame, col: str) -> pd.Series:
    """단일 컬럼 선택이 1-D Series임을 보장(동일 이름 복수 존재 차단)."""
    s = df[col]
    if isinstance(s, pd.DataFrame):
        raise QualityError(f"컬럼 '{col}' 선택이 1-D Series가 아니라 DataFrame입니다.")
    return s


def _assert_no_na(df: pd.DataFrame, cols: Iterable[str]) -> None:
    na_cols = {c: int(df[c].isna().sum()) for c in cols if c in df.columns and df[c].isna().any()}
    if na_cols:
        raise QualityError(f"결측값 존재: {na_cols}")


def _assert_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """비숫자 값(강제 변환 불가) 검출 — 데이터는 변경하지 않음."""
    bad_cols: dict[str, int] = {}
    for c in cols:
        if c in df.columns:
            s = _as_1d_series(df, c)
            s_num = pd.to_numeric(s, errors="coerce")
            n_bad = int(s_num.isna().sum() - s.isna().sum())  # 기존 결측 제외
            if n_bad > 0:
                bad_cols[c] = n_bad
    if bad_cols:
        raise QualityError(f"숫자형 제약 위반(변환 불가 값): {bad_cols}")


def _assert_positive(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            s_num = pd.to_numeric(_as_1d_series(df, c), errors="coerce")
            bad = s_num <= 0
            if bad.any():
                raise QualityError(f"양수 제약 위반: {c} ≤ 0 인 행 {int(bad.sum())}개 (예: {bad.idxmax()})")


def _assert_bar_integrity(df: pd.DataFrame) -> None:
    # low ≤ high, 그리고 low ≤ open/close ≤ high
    cond = (
        (df["low"] <= df["high"])
        & (df["low"] <= df["open"]) & (df["open"] <= df["high"])
        & (df["low"] <= df["close"]) & (df["close"] <= df["high"])
    )
    bad = ~cond
    if bad.any():
        raise QualityError(f"바 무결성 위반(low≤high 및 범위 내 open/close) 행 {int(bad.sum())}개 (예: {bad.idxmax()})")


def validate(
    df: pd.DataFrame,
    *,
    require_volume: bool = False,
    min_rows: int | None = 1,
) -> None:
    """데이터 정합성 검증(원본 수정 없음)."""
    if min_rows is not None and len(df) < min_rows:
        raise QualityError(f"행 수 부족: {len(df)} < {min_rows}")

    # 다차원/중복 컬럼 즉시 실패(초기 단계에서 차단)
    if isinstance(df.columns, pd.MultiIndex) or df.columns.duplicated().any():
        raise QualityError("Duplicate/MultiIndex columns are not allowed. Flatten & dedupe columns first.")

    _assert_columns(df, required=("open", "high", "low", "close"))
    _assert_index_is_utc_dtindex(df)
    _assert_monotonic_strict_and_no_duplicates(df)

    core = ("open", "high", "low", "close")
    _assert_no_na(df, cols=core)
    _assert_numeric(df, cols=core)
    _assert_positive(df, cols=core)
    _assert_bar_integrity(df)

    if require_volume:
        _assert_columns(df, required=("volume",))
    if "volume" in df.columns:
        _assert_no_na(df, cols=("volume",))
        _assert_numeric(df, cols=("volume",))
        s_num = pd.to_numeric(_as_1d_series(df, "volume"), errors="coerce")
        bad = s_num < 0
        if bad.any():
            raise QualityError(f"volume 음수 금지 위반 행 {int(bad.sum())}개 (예: {bad.idxmax()})")


# ------------------------------ 메타 게이트 ------------------------------


def _norm_str(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s or None


def _require_positive_number(meta: Mapping[str, Any], key: str) -> None:
    if key not in meta:
        raise QualityError(f"메타 누락: '{key}'")
    try:
        v = float(meta[key])
    except (TypeError, ValueError):
        raise QualityError(f"메타 '{key}'는 숫자형이어야 합니다.") from None
    if not (v > 0):
        raise QualityError(f"메타 '{key}'는 0보다 커야 합니다. (현재: {v})")


_CALENDAR_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+$")  # 예: XNYS, XKOS, 24x7, CME_GLOBEX 등


def _assert_calendar_meta(meta: Mapping[str, Any], *, require_calendar: bool) -> None:
    cid = _norm_str(meta.get("calendar_id"))
    if require_calendar and not cid:
        raise QualityError("메타 'calendar_id'가 필요합니다.")
    if cid and not _CALENDAR_PATTERN.match(cid):
        raise QualityError(f"메타 'calendar_id' 형식이 올바르지 않습니다: {cid}")


def _assert_fx_meta(meta: Mapping[str, Any], *, require_fx: bool) -> None:
    base_ccy = _norm_str(meta.get("base_currency"))
    local_ccy = _norm_str(meta.get("price_currency") or meta.get("local_currency") or meta.get("currency"))
    fx_src = _norm_str(meta.get("fx_source"))
    fx_ts_raw = _norm_str(meta.get("fx_source_ts"))

    if not base_ccy:
        raise QualityError("메타 'base_currency'가 필요합니다.")

    # 가격 통화가 기준 통화와 다르면 FX 메타 필수
    needs_fx = require_fx and (local_ccy is None or local_ccy != base_ccy)
    if needs_fx:
        if not fx_src:
            raise QualityError("메타 'fx_source'가 필요합니다(비기준 통화 환산).")
        if not fx_ts_raw:
            raise QualityError("메타 'fx_source_ts'가 필요합니다(해당 시점 FX 존재 증빙).")
        fx_ts = pd.to_datetime(fx_ts_raw, utc=True, errors="coerce")
        if pd.isna(fx_ts):
            raise QualityError("메타 'fx_source_ts'를 시각으로 해석할 수 없습니다(ISO8601 권장).")

        # (선택) 스냅샷 기간 내 정합성(있다면 확인)
        start = _norm_str(meta.get("start"))
        end = _norm_str(meta.get("end"))
        if start and end:
            ts_start = pd.to_datetime(start, utc=True, errors="coerce")
            ts_end = pd.to_datetime(end, utc=True, errors="coerce")
            if not (pd.isna(ts_start) or pd.isna(ts_end)):
                if not (ts_start <= fx_ts <= ts_end):
                    raise QualityError(
                        f"메타 'fx_source_ts'가 스냅샷 구간[{ts_start},{ts_end}] 바깥입니다: {fx_ts}"
                    )


def validate_meta(
    meta: Mapping[str, Any],
    *,
    require_fx: bool = True,
    require_calendar: bool = True,
    require_steps: bool = True,
) -> None:
    """
    스냅샷/런 메타 정합성 검증.
    - require_steps: lot_step>0, price_step>0 확인
    - require_calendar: calendar_id 존재 및 형식 확인
    - require_fx: base_currency 필수, (가격통화≠기준통화)면 fx_source/fx_source_ts 필수 및 시점 파싱 확인
    """
    if not isinstance(meta, Mapping):
        raise QualityError("meta는 dict-like Mapping 이어야 합니다.")

    if require_steps:
        _require_positive_number(meta, "lot_step")
        _require_positive_number(meta, "price_step")

    _assert_calendar_meta(meta, require_calendar=require_calendar)
    _assert_fx_meta(meta, require_fx=require_fx)

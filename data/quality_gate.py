# quality_gate.py
"""
정합성(Integrity) 검증 — 공개 API: validate(df, *, require_volume=False, min_rows=1)
- 입력 DataFrame은 수정하지 않으며, 위반 시 QualityError를 즉시 발생시킨다.
- 불변식
  • 인덱스: DatetimeIndex(tz='UTC'), 단조 증가, 중복/음수·0 간격 없음
  • 필수 컬럼: open, high, low, close (숫자형·결측 없음·양수, 1-D Series)
  • 바 무결성: 각 t에서 low ≤ high, low ≤ open/close ≤ high
  • 선택 컬럼: volume 존재 시 숫자형·결측 없음·음수 금지, 1-D Series
"""

from __future__ import annotations

# 표준 라이브러리 우선
from collections.abc import Iterable

# 서드파티
import pandas as pd

__all__ = ["QualityError", "validate"]


class QualityError(ValueError):
    """정합성 위반 예외."""


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
    """단일 컬럼 선택이 1-D Series임을 보장(동일 이름 복수 존재 등 차단)."""
    s = df[col]
    if isinstance(s, pd.DataFrame):
        raise QualityError(f"컬럼 '{col}' 선택이 1-D Series가 아니라 DataFrame입니다(동일 이름 복수 존재 가능).")
    return s


def _assert_no_na(df: pd.DataFrame, cols: Iterable[str]) -> None:
    na_cols = {c: int(df[c].isna().sum()) for c in cols if c in df.columns and df[c].isna().any()}
    if na_cols:
        raise QualityError(f"결측값 존재: {na_cols}")


def _assert_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """비숫자 값(강제 변환 불가)을 검출(데이터는 변경하지 않음)."""
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
            s = _as_1d_series(df, c)
            s_num = pd.to_numeric(s, errors="coerce")
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
    """
    데이터 정합성 검증(원본 수정 없음).
    - require_volume: True면 volume 필수 및 숫자형/결측 없음/음수 금지 검증
    - min_rows: 최소 행 수 요구(None이면 생략)
    """
    if min_rows is not None and len(df) < min_rows:
        raise QualityError(f"행 수 부족: {len(df)} < {min_rows}")

    # 다차원/중복 컬럼 즉시 실패(검증 초기에 추가)
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
        s = _as_1d_series(df, "volume")
        s_num = pd.to_numeric(s, errors="coerce")
        bad = s_num < 0
        if bad.any():
            raise QualityError(f"volume 음수 금지 위반 행 {int(bad.sum())}개 (예: {bad.idxmax()})")

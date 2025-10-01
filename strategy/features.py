# features.py
"""
피처 생성: SMA와 Donchian '이전 N-저가' 기준선.

공개 API
- sma(df, n, *, price_col=None) -> pd.Series
- prev_low_n_tminus1(df, N, *, low_col=None) -> pd.Series
- prev_low_n(df, N, *, low_col=None) -> pd.Series

규약
- *_adj 컬럼이 있으면 우선 사용(close_adj/low_adj 등).
- SMA_n(t)는 윈도 전 구간 NaN, prev_low_N(t-1)은 1칸 시프트 후 롤링 최소.
"""

from __future__ import annotations

# 표준 라이브러리 우선
from typing import Iterable

# 서드파티
import pandas as pd

__all__ = ["FeatureError", "sma", "prev_low_n_tminus1", "prev_low_n"]


class FeatureError(ValueError):
    """피처 산출 단계 예외."""


def _choose_col(df: pd.DataFrame, candidates: Iterable[str], *, param_name: str) -> str:
    """후보 컬럼 중 먼저 존재하는 이름 반환. 없으면 FeatureError."""
    for c in candidates:
        if c in df.columns:
            return c
    raise FeatureError(
        f"required column for '{param_name}' not found; tried={list(candidates)} available={list(df.columns)}"
    )


def _ensure_positive_int(value: int, *, name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise FeatureError(f"'{name}' must be a positive int, got {value!r}")


def _ensure_str_opt(value: str | None, *, name: str) -> None:
    """옵션 문자열 파라미터는 None 또는 단일 str만 허용."""
    if value is not None and not isinstance(value, str):
        raise FeatureError(f"'{name}' must be a single column name (str), not {type(value).__name__}")


def _ensure_1d(x: pd.Series | pd.DataFrame, *, name: str) -> None:
    """입력은 반드시 1-D Series여야 함(중복 컬럼 등으로 2-D 유입 차단)."""
    if isinstance(x, pd.DataFrame) or getattr(x, "ndim", 1) != 1:
        detail = f"DataFrame with columns={list(x.columns)}" if isinstance(x, pd.DataFrame) else f"ndim={getattr(x,'ndim','NA')}"
        raise FeatureError(f"Feature expects a 1-D Series for '{name}', got {detail}. Possible duplicate column names.")


def _to_numeric_series(s: pd.Series | pd.DataFrame, *, name: str) -> pd.Series:
    """1-D Series만 허용, 숫자형 강제. 변환 불가값 존재 시 실패."""
    if isinstance(s, pd.DataFrame):
        raise FeatureError(f"column '{name}' must be a 1-D Series, got DataFrame with columns={list(s.columns)}")
    try:
        return pd.to_numeric(s, errors="raise")
    except Exception as e:
        raise FeatureError(f"column '{name}' must be numeric") from e


def sma(df: pd.DataFrame, n: int, *, price_col: str | None = None) -> pd.Series:
    """단순이동평균(SMA_n). price_col 미지정 시 close_adj→close 우선."""
    _ensure_positive_int(n, name="n")
    _ensure_str_opt(price_col, name="price_col")

    col = price_col or _choose_col(df, ("close_adj", "close"), param_name="price_col")
    s_raw = df[col]
    _ensure_1d(s_raw, name=col)
    s = _to_numeric_series(s_raw, name=col)
    out = s.rolling(window=n, min_periods=n).mean()
    out.name = f"sma{n}"
    return out


def prev_low_n_tminus1(df: pd.DataFrame, N: int, *, low_col: str | None = None) -> pd.Series:
    """Donchian 이전 N-저가: prev_low_N(t-1). low_col 미지정 시 low_adj→low."""
    _ensure_positive_int(N, name="N")
    _ensure_str_opt(low_col, name="low_col")

    col = low_col or _choose_col(df, ("low_adj", "low"), param_name="low_col")
    low_raw = df[col]
    _ensure_1d(low_raw, name=col)
    low = _to_numeric_series(low_raw, name=col)
    out = low.shift(1).rolling(window=N, min_periods=N).min()
    out.name = f"prev_low_{N}_tminus1"
    return out


def prev_low_n(df: pd.DataFrame, N: int, *, low_col: str | None = None) -> pd.Series:
    """Donchian 현재 N-저가: prev_low_N(t). low_col 미지정 시 low_adj→low."""
    _ensure_positive_int(N, name="N")
    _ensure_str_opt(low_col, name="low_col")

    col = low_col or _choose_col(df, ("low_adj", "low"), param_name="low_col")
    low_raw = df[col]
    _ensure_1d(low_raw, name=col)
    low = _to_numeric_series(low_raw, name=col)
    out = low.rolling(window=N, min_periods=N).min()
    out.name = f"prev_low_{N}"
    return out

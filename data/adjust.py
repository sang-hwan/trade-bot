# data/adjust.py
"""
완전 조정(Adj) — 공개 API: apply(df)
- 보정계수: a_t = AdjClose / close
- 조정 OHLC: X_adj = X * a_t (X ∈ {open, high, low, close})
- 산출: open_adj, high_adj, low_adj, close_adj
- AdjClose 미존재 시 a_t = 1.0
- 입력 DataFrame은 변경하지 않음(정렬/보간/삭제 없음)
"""

from __future__ import annotations

import pandas as pd

__all__ = ["AdjustError", "apply"]


class AdjustError(ValueError):
    """조정(Adj) 단계 예외."""


def _require_cols(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise AdjustError(f"필수 컬럼 누락: {missing}")


def apply(
    df: pd.DataFrame,
    *,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    adjclose_col: str = "AdjClose",
) -> pd.DataFrame:
    """
    조정 컬럼(*_adj)을 추가해 새 DataFrame 반환.
    예외:
      - AdjClose 존재 시 결측/비양수(≤0) → 실패
      - close 결측/비양수(≤0) → 실패(분모 안전성)
      - OHLC 컬럼이 1-D 가드 위반(DataFrame 반환) → 실패
    """
    _require_cols(df, (open_col, high_col, low_col, close_col))

    # 1-D 가드: 동일 이름 중복 등으로 df[c]가 DataFrame이 되는 경우 차단
    for c in (open_col, high_col, low_col, close_col):
        s = df[c]
        if getattr(s, "ndim", 1) != 1:
            raise AdjustError(f"Column '{c}' must be 1-D (got DataFrame). Check duplicate column names.")

    out = df.copy()

    if adjclose_col in out.columns:
        adj = pd.to_numeric(out[adjclose_col], errors="coerce")
        close = pd.to_numeric(out[close_col], errors="coerce")
        if adj.isna().any():
            raise AdjustError(f"{adjclose_col}에 결측값이 존재합니다.")
        if (adj <= 0).any():
            raise AdjustError(f"{adjclose_col}에 비양수(≤0) 값이 존재합니다.")
        if close.isna().any() or (close <= 0).any():
            raise AdjustError(f"{close_col}가 결측이거나 비양수(≤0)입니다.")
        a = adj / close
    else:
        a = pd.Series(1.0, index=out.index, dtype="float64")

    out["open_adj"] = pd.to_numeric(out[open_col], errors="coerce") * a
    out["high_adj"] = pd.to_numeric(out[high_col], errors="coerce") * a
    out["low_adj"] = pd.to_numeric(out[low_col], errors="coerce") * a
    out["close_adj"] = pd.to_numeric(out[close_col], errors="coerce") * a

    for c in ("open_adj", "high_adj", "low_adj", "close_adj"):
        out[c] = out[c].astype("float64")

    return out

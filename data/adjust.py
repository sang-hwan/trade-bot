# adjust.py
"""
완전 조정(Adj) 모듈 — 공개 API: apply(df)
- 보정계수: a_t = AdjClose / close
- 조정 OHLC: X_adj = X * a_t (X ∈ {open, high, low, close})
- 산출 컬럼: open_adj, high_adj, low_adj, close_adj
- AdjClose 미존재 시 a_t=1로 간주하여 *_adj를 원본과 동일하게 생성
- 입력 DataFrame은 수정하지 않으며 정렬/보간/삭제를 수행하지 않음
"""

from __future__ import annotations

import pandas as pd

__all__ = ["AdjustError", "apply"]


class AdjustError(ValueError):
    """조정(Adj) 실패 시 발생하는 예외."""


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
    입력 df에 조정 컬럼(*_adj)을 추가하여 새 DataFrame을 반환.
    예외:
      - AdjClose 존재 시 결측/비양수(≤0) 값 → 실패
      - close 결측/비양수(≤0) → 실패  (분모 안전성)
    """
    _require_cols(df, (open_col, high_col, low_col, close_col))
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
        # 암호화폐 등 AdjClose 미제공 자산: a_t = 1.0
        a = pd.Series(1.0, index=out.index, dtype="float64")

    out["open_adj"] = pd.to_numeric(out[open_col], errors="coerce") * a
    out["high_adj"] = pd.to_numeric(out[high_col], errors="coerce") * a
    out["low_adj"] = pd.to_numeric(out[low_col], errors="coerce") * a
    out["close_adj"] = pd.to_numeric(out[close_col], errors="coerce") * a

    # 타입 고정: float64
    for c in ("open_adj", "high_adj", "low_adj", "close_adj"):
        out[c] = out[c].astype("float64")

    return out

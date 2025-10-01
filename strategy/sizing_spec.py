# sizing_spec.py
"""
전략/사이징 스펙: Fixed Fractional **스펙만** 산출한다(수량/체결 계산 금지).

- 타이밍(롱): 엔트리 Open(t)에서 사용할 스탑 레벨은 `prev_low_N(t-1)`
- 출력: index=df.index, columns=['f','stop_level','lot_step','V','PV']
  * f: 위험 비율 (0 < f ≤ 1)
  * stop_level: prev_low_N(t-1)  # 결정=Close(t-1), 집행=Open(t) 규약과 합치
  * lot_step: 최소 수량/호가 단위 s (> 0)
  * V: 선물 승수(필요 시), PV: FX pip 값(필요 시) — 없으면 <NA>

공개 API
- build_fixed_fractional_spec(df, *, N, f, lot_step, low_col=None, V=None, PV=None) -> pd.DataFrame
"""

from __future__ import annotations

import pandas as pd

from .features import prev_low_n_tminus1

__all__ = ["SizingSpecError", "build_fixed_fractional_spec"]


class SizingSpecError(ValueError):
    """사이징 스펙 산출 단계 예외."""


def _ensure_pos_int(value: int, *, name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise SizingSpecError(f"'{name}' must be a positive int, got {value!r}")


def _ensure_frac(value: float, *, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as e:
        raise SizingSpecError(f"'{name}' must be a float-like number, got {value!r}") from e
    if not (0.0 < v <= 1.0):
        raise SizingSpecError(f"'{name}' must be in (0, 1], got {value!r}")
    return v


def _ensure_pos_num(value: float, *, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as e:
        raise SizingSpecError(f"'{name}' must be a float-like number, got {value!r}") from e
    if v <= 0.0:
        raise SizingSpecError(f"'{name}' must be > 0, got {value!r}")
    return v


def build_fixed_fractional_spec(
    df: pd.DataFrame,
    *,
    N: int,
    f: float,
    lot_step: float | int,
    low_col: str | None = None,
    V: float | None = None,
    PV: float | None = None,
) -> pd.DataFrame:
    """
    Fixed Fractional 사이징 스펙 시계열을 생성한다.

    - stop_level = prev_low_N(t-1) (룩어헤드 금지, 시뮬레이터 On-Open 계산 전제)
    - 수량(Q)·라운딩·수수료/슬리피지는 시뮬레이션 단계 책임(여기서 계산하지 않음).
    """
    _ensure_pos_int(N, name="N")
    f_val = _ensure_frac(f, name="f")
    lot_val = _ensure_pos_num(lot_step, name="lot_step")

    stop_level = prev_low_n_tminus1(df, N, low_col=low_col).astype("float64")

    idx = df.index
    out = pd.DataFrame(
        {
            "f": pd.Series(f_val, index=idx, dtype="float64"),
            "stop_level": stop_level,
            "lot_step": pd.Series(lot_val, index=idx, dtype="float64"),
            "V": pd.Series(pd.NA if V is None else float(V), index=idx, dtype="Float64"),
            "PV": pd.Series(pd.NA if PV is None else float(PV), index=idx, dtype="Float64"),
        }
    )
    return out

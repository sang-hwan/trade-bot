# signals.py
"""
전략/신호 단계: SMA10/50 크로스(롱 온리) 신호 생성.

- SSOT: 데이터 정합화·스냅샷·시뮬레이션은 포함하지 않음.
- 입력 규약: *_adj 컬럼이 있으면 SMA 계산 시 항상 우선 사용(`close_adj` 등).
- 타이밍 규약:
  결정(Decision)=Close(t-1) 확정 특징, 집행(Execution)=Open(t)
  → 반환 시리즈 `signal_next[t]`는 t 시가(Open) 집행 후보를 의미.

공개 API
- sma_cross_long_only(df, *, short=10, long=50, epsilon=0.0, price_col=None) -> pd.Series
  (name='signal_next', dtype=Int8, 값 {1,0,<NA>})
"""

from __future__ import annotations

import pandas as pd

from .features import sma

__all__ = ["SignalError", "sma_cross_long_only"]


class SignalError(ValueError):
    """신호 산출 단계 예외."""


def _ensure_positive_int(value: int, *, name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise SignalError(f"'{name}' must be a positive int, got {value!r}")


def _ensure_nonneg_float(value: float, *, name: str) -> None:
    try:
        v = float(value)
    except (TypeError, ValueError) as e:
        raise SignalError(f"'{name}' must be a float-like number, got {value!r}") from e
    if v < 0.0:
        raise SignalError(f"'{name}' must be ≥ 0, got {value!r}")


def sma_cross_long_only(
    df: pd.DataFrame,
    *,
    short: int = 10,
    long: int = 50,
    epsilon: float = 0.0,
    price_col: str | None = None,
) -> pd.Series:
    """
    SMA(short) − SMA(long)와 타이브레이크 ε(≥0)에 기반한 롱 온리 신호를 반환한다.

    정의(타이브레이크):

    $$
    \\texttt{signal\\_next} =
    \\begin{cases}
    1 & \\text{if } \\mathrm{SMA}_{\\text{short}} - \\mathrm{SMA}_{\\text{long}} > \\epsilon \\\\
    0 & \\text{otherwise}
    \\end{cases}
    \\qquad (\\epsilon \\ge 0)
    $$

    타임라인:
    판단은 Close(t−1), 집행은 Open(t). `signal_next[t]=1`이면 t 시가 매수 집행 후보.
    """
    _ensure_positive_int(short, name="short")
    _ensure_positive_int(long, name="long")
    _ensure_nonneg_float(epsilon, name="epsilon")

    # 피처 계산(입력 df는 수정하지 않음)
    s_short = sma(df, short, price_col=price_col)
    s_long = sma(df, long, price_col=price_col)

    # diff > ε 판단은 결정 시점(t-1)에 존재해야 하므로, 집행 시점 정렬을 위해 +1 시프트 적용
    diff = s_short - s_long
    decided = (diff > float(epsilon)).astype("boolean")  # True/False/NA
    signal_next = decided.shift(1).astype("Int8")  # {1,0,<NA>}
    signal_next.name = "signal_next"
    return signal_next

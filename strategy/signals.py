# strategy/signals.py
"""
전략/신호: SMA10/50 크로스(롱 온리) 신호 생성.

타이밍 규약
- 결정(Decision)=Close(t-1), 집행(Execution)=Open(t)
- 반환 Series `signal_next[t]`는 t 시가(Open) 집행 후보를 의미

공개 API
- sma_cross_long_only(
    df, *,
    short=10, long=50, epsilon=0.0, price_col=None,
    anchor_time=None, anchor_tz="Asia/Seoul"
  ) -> pd.Series
  - anchor_time: 24x7 자산(코인)용 "가상 데일리 경계" 시각("HH:MM"), 연산은 기존 SMA와 동일
"""

from __future__ import annotations

import re
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


def _validate_anchor(anchor_time: str | None, anchor_tz: str | None) -> None:
    """가상 데일리 경계 파라미터 형식 검증."""
    if anchor_time is None:
        return
    if not isinstance(anchor_time, str) or not re.fullmatch(r"\d{2}:\d{2}", anchor_time):
        raise SignalError(f"'anchor_time' must be 'HH:MM' like '09:00', got {anchor_time!r}")
    if not anchor_tz or not isinstance(anchor_tz, str):
        raise SignalError(f"'anchor_tz' must be a non-empty string, got {anchor_tz!r}")


def sma_cross_long_only(
    df: pd.DataFrame,
    *,
    short: int = 10,
    long: int = 50,
    epsilon: float = 0.0,
    price_col: str | None = None,
    anchor_time: str | None = None,   # 코인용 가상 데일리 경계(예: "09:00")
    anchor_tz: str = "Asia/Seoul",    # 경계 기준 타임존
) -> pd.Series:
    """
    SMA(short) − SMA(long) > ε(ε≥0)이면 1, 아니면 0을 다음 시점에 제안한다.
    판단은 Close(t−1), 집행은 Open(t).
    """
    _ensure_positive_int(short, name="short")
    _ensure_positive_int(long, name="long")
    _ensure_nonneg_float(epsilon, name="epsilon")
    _validate_anchor(anchor_time, anchor_tz)

    s_short = sma(df, short, price_col=price_col)
    s_long = sma(df, long, price_col=price_col)

    diff = s_short - s_long
    decided = (diff > float(epsilon)).astype("boolean")
    signal_next = decided.shift(1).astype("Int8")
    signal_next.name = "signal_next"

    # 메타(연산 동일, 정보만 제공)
    signal_next.attrs = {
        "anchor_time": anchor_time,
        "anchor_tz": anchor_tz,
        "short": short,
        "long": long,
        "epsilon": float(epsilon),
        "price_col": price_col,
    }
    return signal_next

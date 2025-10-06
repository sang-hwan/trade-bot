# strategy/stops.py
"""
전략/스탑: Donchian N-저가 이탈(롱 기준) 스탑 판정과 레벨을 산출.

- SSOT: 데이터 정합/스냅샷/시뮬레이션 로직은 포함하지 않음.
- 입력 규약: `low_adj`가 있으면 우선 사용, 없으면 `low`.
- 타이밍: 판정=Close(t)에서  L_t ≤ prev_low_N(t−1)  → 집행=Open(t+1)
  * 반환: `stop_hit[t]`는 t 종가 기준 **다음 시가(Open t+1) 청산 후보**
  * `stop_level[t] = prev_low_N(t)` (집행 시점의 기준선)

공개 API
- donchian_stop_long(df, N: int, *, low_col: str | None = None) -> pd.DataFrame
  (columns: 'stop_hit' [pandas nullable boolean], 'stop_level' [float])
"""

from __future__ import annotations

import pandas as pd

from .features import prev_low_n_tminus1, prev_low_n

__all__ = ["StopError", "donchian_stop_long"]


class StopError(ValueError):
    """스탑 산출 단계 예외."""


def _ensure_positive_int(value: int, *, name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise StopError(f"'{name}' must be a positive int, got {value!r}")


def _pick_low_col(df: pd.DataFrame, low_col: str | None) -> str:
    """`low_adj` → `low` 우선순위로 실제 사용할 low 컬럼명을 결정."""
    if low_col is not None:
        return low_col
    if "low_adj" in df.columns:
        return "low_adj"
    return "low"


def donchian_stop_long(
    df: pd.DataFrame, N: int, *, low_col: str | None = None
) -> pd.DataFrame:
    """
    Donchian N-저가 이탈(롱 기준) 스탑을 계산한다.

    판정 수식:
    $$ L_t \\le \\mathrm{prev\\_low}_N(t-1) $$

    레벨:
    $$ \\texttt{stop\\_level}(t) = \\mathrm{prev\\_low}_N(t) $$

    - 초기 t < N 구간은 판단 불가로 <NA>, 레벨은 NaN.
    """
    _ensure_positive_int(N, name="N")

    # 기준선(SSOT: features 사용)
    prev_low_tm1 = prev_low_n_tminus1(df, N, low_col=low_col)  # 판정 기준선 (t-1)
    level_t = prev_low_n(df, N, low_col=low_col)               # 집행용 레벨 (t)

    # 비교 대상 L_t 확보
    col = _pick_low_col(df, low_col)
    if col not in df.columns:
        raise StopError(f"column '{col}' not found in df")
    low = pd.to_numeric(df[col], errors="raise")

    # 판정: L_t ≤ prev_low_N(t-1)
    # - 둘 중 하나라도 NaN이면 판단 <NA> 유지
    mask_valid = low.notna() & prev_low_tm1.notna()
    decided = (low <= prev_low_tm1)
    stop_hit = decided.where(mask_valid).astype("boolean")  # {True, False, <NA>}
    stop_hit.name = "stop_hit"

    out = pd.DataFrame({"stop_hit": stop_hit, "stop_level": level_t})
    return out

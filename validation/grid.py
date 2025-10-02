# grid.py
"""
민감도/스트레스 파라미터 그리드 생성기.

공개 API
- grid_from_spaces(spaces: Mapping[str, Sequence], *, max_size: int|None=50_000, validate: bool=True) -> list[dict]
- linspace(start: float, stop: float, num: int, *, endpoint: bool=True, round_to: int|None=None) -> list[float]
- arange(start: float, stop: float, step: float, *, endpoint: bool=True, round_to: int|None=None) -> list[float]
- logspace(start_exp: float, stop_exp: float, num: int, *, base: float=10.0, endpoint: bool=True, round_to: int|None=None) -> list[float]

계약
- 입력은 {파라미터명: 값 시퀀스}(빈 시퀀스 금지).
- 반환은 결정적 순서(파라미터명 알파벳 정렬)의 조합 리스트.
- validate=True면 기본 유효성 검사(n/N/sma_*: 양의 정수, f∈(0,1], commission/slip/epsilon ≥0, lot_step>0).
"""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence
import math

__all__ = ["grid_from_spaces", "linspace", "arange", "logspace"]


# ---------- 수열 유틸 ----------

def _round_if(x: float, ndigits: int | None) -> float:
    return round(x, ndigits) if ndigits is not None else x


def linspace(
    start: float,
    stop: float,
    num: int,
    *,
    endpoint: bool = True,
    round_to: int | None = None,
) -> list[float]:
    """등간격 수열(numpy 미사용)."""
    if num <= 0:
        raise ValueError("num must be positive.")
    if num == 1:
        return [_round_if(stop if endpoint else start, round_to)]
    steps = num - 1 if endpoint else num
    step = (stop - start) / steps
    out = [start + i * step for i in range(num)]
    if endpoint:
        out[-1] = stop
    return [_round_if(v, round_to) for v in out]


def arange(
    start: float,
    stop: float,
    step: float,
    *,
    endpoint: bool = True,
    round_to: int | None = None,
) -> list[float]:
    """등차 수열(부동소수 오차 보정)."""
    if step <= 0:
        raise ValueError("step must be positive.")
    n = max(0, math.floor((stop - start) / step + 1e-12))
    out = [start + i * step for i in range(n + 1)]
    if endpoint:
        last = start + (n + 1) * step
        if last <= stop + 1e-12:
            out.append(last)
    return [_round_if(v, round_to) for v in out]


def logspace(
    start_exp: float,
    stop_exp: float,
    num: int,
    *,
    base: float = 10.0,
    endpoint: bool = True,
    round_to: int | None = None,
) -> list[float]:
    """로그 등간격: base**exp, exp는 등간격."""
    exps = linspace(start_exp, stop_exp, num, endpoint=endpoint)
    out = [base ** e for e in exps]
    return [_round_if(v, round_to) for v in out]


# ---------- 검증/그리드 ----------

def _is_pos_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x > 0


def _is_nonneg_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and float(x) >= 0.0


def _is_pos_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and float(x) > 0.0


def _validate_param(name: str, value: Any) -> None:
    """핵심 파라미터 기본 유효성(실패 시 ValueError)."""
    if name in {"n", "N", "sma_short", "sma_long"}:
        if not _is_pos_int(value):
            raise ValueError(f"{name} must be a positive int.")
    elif name == "f":
        if not (isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 < float(value) <= 1.0):
            raise ValueError("f must be in (0, 1].")
    elif name in {"commission", "commission_rate", "slip", "epsilon"}:
        if not _is_nonneg_number(value):
            raise ValueError(f"{name} must be a non-negative number.")
    elif name == "lot_step":
        if not _is_pos_number(value):
            raise ValueError("lot_step must be > 0.")
    elif name in {"V", "PV"}:
        if value is not None and not _is_pos_number(value):
            raise ValueError(f"{name} must be > 0 if provided.")
    # 기타 파라미터는 호출 측 계약에 위임.


def _normalize_space(values: Sequence[Any]) -> list[Any]:
    """시퀀스를 리스트로 정규화(빈 시퀀스 금지)."""
    vals = list(values)
    if not vals:
        raise ValueError("Parameter space contains an empty sequence.")
    return vals


def grid_from_spaces(
    spaces: Mapping[str, Sequence[Any]],
    *,
    max_size: int | None = 50_000,
    validate: bool = True,
) -> list[dict[str, Any]]:
    """
    파라미터 공간의 데카르트 곱.
    - 결정적 순서: 파라미터명 알파벳 정렬.
    - max_size 초과 시 ValueError(조합 폭발 방지).
    """
    if not spaces:
        raise ValueError("spaces must not be empty.")

    keys = sorted(spaces.keys())
    norm_spaces = [_normalize_space(spaces[k]) for k in keys]

    size = 1
    for seq in norm_spaces:
        size *= len(seq)
        if max_size is not None and size > max_size:
            raise ValueError(f"Grid too large ({size} combos). Reduce spaces or raise max_size.")

    combos: list[dict[str, Any]] = []
    for tpl in product(*norm_spaces):
        combo = {k: v for k, v in zip(keys, tpl)}
        if validate:
            for k, v in combo.items():
                _validate_param(k, v)
        combos.append(combo)
    return combos

# simulation/sizing_on_open.py
"""
사이징 계산(온-오픈 전용, 룩어헤드 금지).

공개 API
- calc_distance_long(entry_price, stop_level) -> float
- floor_step(q, step) -> float
- size_from_spec(entry_price, equity, *, f, stop_level, lot_step, V=None, PV=None) -> dict[str, float]
- anchor_open_price(df, *, anchor_time="09:00", anchor_tz="Asia/Seoul", price_col=None) -> float
  * 코인처럼 실개장이 없는 24x7 자산의 "가상 데일리 경계"(지정 시각 첫 체결가)를 산출해 entry_price로 사용 가능.

규칙
- R=f·E, D=Entry-stop_level(롱)
- Q_raw=R/denom, Q_exec=⌊Q_raw/lot_step⌋·lot_step
- denom = D(현물) | D·V(선물) | D_pips·PV(FX)

예외 처리
- 잘못된 입력은 0.0/빈 값으로 귀결시켜 후속 계산 안전성을 확보한다.
"""

from __future__ import annotations

from math import floor, isfinite
import pandas as pd

__all__ = ["calc_distance_long", "floor_step", "size_from_spec", "anchor_open_price"]


def calc_distance_long(entry_price: float, stop_level: float) -> float:
    """롱 기준 스탑 거리: D = max(entry - stop_level, 0.0). 비정상 값은 0.0."""
    try:
        d = float(entry_price) - float(stop_level)
    except (TypeError, ValueError):
        return 0.0
    if not isfinite(d) or d <= 0.0:
        return 0.0
    return d


def floor_step(q: float, step: float) -> float:
    """로트/호가 단위 하향 라운딩: ⌊q/step⌋·step (step<=0 또는 q<=0이면 0.0)."""
    try:
        qf = float(q)
        sf = float(step)
    except (TypeError, ValueError):
        return 0.0
    if sf <= 0.0 or qf <= 0.0 or (not isfinite(qf)) or (not isfinite(sf)):
        return 0.0
    return floor(qf / sf) * sf


def _denominator(D: float, V: float | None, PV: float | None) -> float:
    """분모: 선물 D·V > FX D·PV > 현물 D. D 비정상이면 0.0."""
    if not isfinite(D) or D <= 0.0:
        return 0.0
    if V is not None:
        try:
            return D * float(V)
        except (TypeError, ValueError):
            return 0.0
    if PV is not None:
        try:
            return D * float(PV)
        except (TypeError, ValueError):
            return 0.0
    return D


def size_from_spec(
    entry_price: float,
    equity: float,
    *,
    f: float,
    stop_level: float,
    lot_step: float,
    V: float | None = None,
    PV: float | None = None,
) -> dict[str, float]:
    """
    Fixed Fractional 스펙으로부터 온-오픈 수량 산출(계산만).
    반환: {"Q_raw","Q_exec","R","D","denom","lot_step"}
    """
    try:
        E = float(equity)
        fval = float(f)
    except (TypeError, ValueError):
        return {"Q_raw": 0.0, "Q_exec": 0.0, "R": 0.0, "D": 0.0, "denom": 0.0, "lot_step": 0.0}

    try:
        lot_step_f = float(lot_step)
    except (TypeError, ValueError):
        lot_step_f = 0.0
    lot_step_f = lot_step_f if (isfinite(lot_step_f) and lot_step_f > 0.0) else 0.0

    R = fval * E
    D = calc_distance_long(entry_price, stop_level)
    denom = _denominator(D, V, PV)

    if R <= 0.0 or denom <= 0.0 or not (isfinite(R) and isfinite(denom)):
        return {"Q_raw": 0.0, "Q_exec": 0.0, "R": float(R), "D": float(D), "denom": float(denom), "lot_step": lot_step_f}

    Q_raw = R / denom
    Q_exec = floor_step(Q_raw, lot_step_f)

    return {
        "Q_raw": float(Q_raw),
        "Q_exec": float(Q_exec),
        "R": float(R),
        "D": float(D),
        "denom": float(denom),
        "lot_step": lot_step_f,
    }


# ── 24x7 자산용 앵커 오픈가 ───────────────────────────────────────────────────

def anchor_open_price(
    df: pd.DataFrame,
    *,
    anchor_time: str = "09:00",
    anchor_tz: str = "Asia/Seoul",
    price_col: str | None = None,
) -> float:
    """
    24x7 시계열에서 '가상 데일리 경계(anchor_tz 기준 HH:MM)'의 첫 체결가를 반환.
    - df.index: DatetimeIndex. tz-naive면 UTC로 간주.
    - 가격 우선순위: 'open' > price_col > 'close'.
    - 최신 레코드 시각을 기준으로 해당/직전 일 경계 구간에서 첫 레코드의 가격을 사용.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return 0.0

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df_local = df.copy()
    df_local.index = idx.tz_convert(anchor_tz)

    # 가격 컬럼 선택
    candidates: tuple[str, ...] = ("open",) + ((price_col,) if price_col else tuple()) + ("close",)
    col = next((c for c in candidates if c in df_local.columns), None)
    if col is None:
        return 0.0

    try:
        hh, mm = (int(x) for x in anchor_time.split(":"))
    except (ValueError, AttributeError):
        return 0.0

    last_local = df_local.index.max()
    anchor_today = last_local.replace(hour=hh, minute=mm, second=0, microsecond=0)

    if last_local.time() >= anchor_today.time():
        start = anchor_today
    else:
        start = anchor_today - pd.Timedelta(days=1)

    end = start + pd.Timedelta(days=1)

    mask = (df_local.index >= start) & (df_local.index < end)
    if not mask.any():
        return 0.0

    first_row = df_local.loc[mask].iloc[0]
    try:
        price = float(first_row[col])
    except (TypeError, ValueError):
        return 0.0
    return price if isfinite(price) and price > 0.0 else 0.0

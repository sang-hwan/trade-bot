"""
Donchian 스탑 엔진: OHLC 품질 게이트 적용 후 전일 기준 최저가 임계값 L_N(t-1)을 배치/스트리밍으로 계산하고,
룩어헤드 없이 스탑 트리거 및 기본 체결가(NEXT-OPEN)를 산출합니다.

이 파일의 목적:
- 데이터 품질 검증(양수·OHLC 질서·정렬·중복 제거)과 Donchian 임계값 계산 경로(배치/스트리밍)를 표준화합니다.
- 스탑 규칙 L_t ≤ L_N(t-1)을 일관 적용하고, 스탑 히트 시 기본 체결 정책(NEXT-OPEN)을 함께 제공합니다.

사용되는 변수와 함수 목록:
- 변수
  - 없음

- 함수
  - quality_gate_ohlc(df: pandas.DataFrame, open_col: str = "open", high_col: str = "high", low_col: str = "low", close_col: str = "close")
    - 역할: OHLC 품질 게이트(정렬·중복 제거·NaN 제거·양수·OHLC 질서 제약) 적용
    - 입력값: df — 원본 OHLC 데이터프레임(인덱스는 시간), open/high/low/close 컬럼명
    - 반환값: pandas.DataFrame — 품질 기준을 통과한 정렬·정제 데이터

  - donchian_prev_low_batch(lows: pandas.Series, n: int)
    - 역할: 배치 방식으로 L_N(t-1) 시퀀스 계산(rolling min + 1칸 시프트)
    - 입력값: lows — 저가 시계열, n>0 — 창 길이
    - 반환값: pandas.Series — 각 t에서의 L_N(t-1); 초기 n 구간은 NaN

  - donchian_prev_low_stream(lows: Iterable[float], n: int)
    - 역할: 모노토닉 덱으로 L_N(t-1) 시퀀스 계산(O(1) 기대)
    - 입력값: lows — 저가 이터러블, n>0 — 창 길이
    - 반환값: list[float|None] — 각 t에서의 L_N(t-1); 초기 n 구간은 None

  - donchian_stop_signals(df: pandas.DataFrame, n: int, low_col: str = "low", open_col: str = "open", high_col: str = "high", close_col: str = "close", use_stream: bool = False)
    - 역할: 품질 게이트 → L_N(t-1) 산출(배치/스트리밍 선택) → 스탑 트리거와 NEXT-OPEN 체결가 산출
    - 입력값: df — OHLC 데이터, n>0 — 창 길이, *_col — 컬럼명, use_stream — True면 스트리밍 경로 사용
    - 반환값: pandas.DataFrame — {"prev_low": L_N(t-1), "stop_hit": bool, "fill_next_open": 체결가 또는 NaN}

  - check_prev_low_equivalence(lows: pandas.Series, n: int, atol: float = 0.0)
    - 역할: 배치/스트리밍 prev_low 동치성 검증
    - 입력값: lows — 저가 시계열, n>0 — 창 길이, atol≥0 — 허용 오차
    - 반환값: bool — 두 계산 경로가 동일(또는 오차 내)하면 True

파일의 흐름(→ / ->):
- 원본 OHLC → quality_gate_ohlc()로 정제 → donchian_prev_low_*()로 L_N(t-1) 계산
  -> (low_t ≤ L_N(t-1)) 스탑 판정 → stop_hit=True 인 시점의 다음 봉 시가로 fill_next_open 산출(NEXT-OPEN 체결) → 결과 데이터프레임 반환
"""

from collections import deque
from collections.abc import Iterable
from typing import List, Optional
import pandas as pd


def quality_gate_ohlc(
    df: pd.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    cols = [open_col, high_col, low_col, close_col]
    df = df.copy()
    df = df.sort_index(kind="mergesort")
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=cols)
    pos_mask = (df[cols] > 0).all(axis=1)
    ord_mask = (
        (df[low_col] <= df[open_col])
        & (df[open_col] <= df[high_col])
        & (df[low_col] <= df[close_col])
        & (df[close_col] <= df[high_col])
        & (df[low_col] <= df[high_col])
    )
    df = df[pos_mask & ord_mask]
    return df


def donchian_prev_low_batch(lows: pd.Series, n: int) -> pd.Series:
    return lows.shift(1).rolling(window=n, min_periods=n).min()


def donchian_prev_low_stream(lows: Iterable[float], n: int) -> List[Optional[float]]:
    deq = deque()
    out: List[Optional[float]] = []
    for t, x in enumerate(lows):
        if t >= n:
            out.append(deq[0][1] if deq else None)
        else:
            out.append(None)
        while deq and deq[-1][1] >= x:
            deq.pop()
        deq.append((t, float(x)))
        earliest = t - n + 1
        while deq and deq[0][0] < earliest:
            deq.popleft()
    return out


def donchian_stop_signals(
    df: pd.DataFrame,
    n: int,
    low_col: str = "low",
    open_col: str = "open",
    high_col: str = "high",
    close_col: str = "close",
    use_stream: bool = False,
) -> pd.DataFrame:
    df = quality_gate_ohlc(df, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)
    lows = df[low_col].astype(float)
    if use_stream:
        prev_low_vals = donchian_prev_low_stream(lows.values.tolist(), n)
        prev_low = pd.Series(prev_low_vals, index=df.index, dtype="float64")
    else:
        prev_low = donchian_prev_low_batch(lows, n)
    stop_hit = (lows <= prev_low) & prev_low.notna()
    next_open = df[open_col].shift(-1)
    fill_next_open = next_open.where(stop_hit)
    out = pd.DataFrame(
        {
            "prev_low": prev_low,
            "stop_hit": stop_hit,
            "fill_next_open": fill_next_open,
        },
        index=df.index,
    )
    return out


def check_prev_low_equivalence(lows: pd.Series, n: int, atol: float = 0.0) -> bool:
    a = donchian_prev_low_batch(lows, n)
    b = pd.Series(donchian_prev_low_stream(lows.values.tolist(), n), index=lows.index, dtype="float64")
    mask = a.notna() & b.notna()
    if not mask.any():
        return True
    if atol == 0.0:
        return bool((a[mask] == b[mask]).all())
    diff = (a[mask] - b[mask]).abs()
    return bool((diff <= atol).all())

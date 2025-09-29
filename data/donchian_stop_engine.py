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
    *,
    assume_gated: bool = False,
) -> pd.DataFrame:
    if not assume_gated:
        df = quality_gate_ohlc(df, open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)

    if low_col == "low" and "low_adj" in df.columns:
        low_col = "low_adj"
    if open_col == "open" and "open_adj" in df.columns:
        open_col = "open_adj"

    lows = df[low_col].astype(float)

    if use_stream:
        prev_low_vals = donchian_prev_low_stream(lows.values.tolist(), n)
        prev_low = pd.Series(prev_low_vals, index=df.index, dtype="float64")
    else:
        prev_low = donchian_prev_low_batch(lows, n)

    stop_hit = (lows <= prev_low) & prev_low.notna()
    next_open = df[open_col].shift(-1)
    fill_next_open = next_open.where(stop_hit)

    return pd.DataFrame(
        {"prev_low": prev_low, "stop_hit": stop_hit, "fill_next_open": fill_next_open},
        index=df.index,
    )


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

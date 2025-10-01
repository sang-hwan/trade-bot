# collect.py
"""
원천(API) → DatetimeIndex[UTC] → 컬럼 표준화(open, high, low, close)
- 주식: Yahoo Finance (yfinance)
- 코인: Upbit Public API

출력
- 인덱스: pandas.DatetimeIndex(tz='UTC'), 오름차순, 중복 제거
- 필수 컬럼: open, high, low, close (float)
- 선택 컬럼: volume (float), AdjClose (주식)
"""

from __future__ import annotations

# 표준 라이브러리
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# 서드파티
import pandas as pd

# Upbit(지연 의존)
try:
    import requests as _requests  # type: ignore
except Exception:
    _requests = None

# Yahoo (yfinance)
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

__all__ = ["CollectResult", "collect", "fetch_yahoo_ohlcv", "fetch_upbit_ohlcv"]

# ---------- 공통 유틸 ----------

def _as_utc_ts(dt_like: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(dt_like)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _to_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_localize(timezone.utc) if idx.tz is None else idx.tz_convert(timezone.utc)


def _parse_dt(dt: str | None) -> datetime | None:
    if not dt:
        return None
    ts = pd.to_datetime(dt, utc=True)
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime()
    raise ValueError("start/end는 단일 시각 문자열이어야 합니다.")


def _flatten_and_dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """MultiIndex → 단일 레벨, 이후 중복 컬럼은 마지막 것만 유지."""
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = [c[0] for c in df.columns]
        if any(name in lvl0 for name in ("Open", "High", "Low", "Close", "Adj Close", "Volume")):
            df = df.copy(); df.columns = lvl0
        else:
            df = df.copy(); df.columns = [c[-1] for c in df.columns]
    if getattr(df.columns, "duplicated", None) is not None and df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """인덱스 정리 및 표준 컬럼 순서."""
    if df.empty:
        return df
    df = df[~df.index.duplicated(keep="last")].sort_index()
    col_order = ["open", "high", "low", "close", "volume", "AdjClose"]
    cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
    return df[cols]

# ---------- Yahoo (Stocks) ----------

def fetch_yahoo_ohlcv(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """yfinance 사용(auto_adjust=False; AdjClose는 이후 단계에서 사용)."""
    if yf is None:
        raise RuntimeError("yfinance가 필요합니다. `pip install yfinance` 후 실행하세요.")

    dt_start = _parse_dt(start)
    dt_end = _parse_dt(end)

    dl = yf.download(
        tickers=symbol,
        start=dt_start,
        end=(dt_end + timedelta(days=1)) if dt_end else None,  # yfinance end 보정
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if dl.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "AdjClose"])

    dl = _flatten_and_dedupe_columns(dl)
    dl = dl.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "AdjClose",
            "Volume": "volume",
        }
    )
    dl = _flatten_and_dedupe_columns(dl)
    dl.index = _to_utc_index(dl.index)

    if dt_start:
        dl = dl[dl.index >= _as_utc_ts(dt_start)]
    if dt_end:
        dl = dl[dl.index <= _as_utc_ts(dt_end)]

    for c in ("open", "high", "low", "close", "volume", "AdjClose"):
        if c in dl.columns:
            dl[c] = pd.to_numeric(dl[c], errors="coerce")

    dl = dl[~dl.index.duplicated(keep="last")].sort_index()
    return _finalize(dl)

# ---------- Upbit (Crypto) ----------

_UPBIT_BASE = "https://api.upbit.com/v1/candles"

def _upbit_interval_path(interval: str) -> str:
    """Upbit interval 문자열 → API path."""
    s = interval.strip().lower()
    if s.endswith("m"):
        unit = int(s[:-1])
        if unit not in (1, 3, 5, 10, 15, 30, 60, 240):
            raise ValueError("Upbit 분봉은 {1,3,5,10,15,30,60,240}만 지원합니다.")
        return f"minutes/{unit}"
    if s in ("1d", "d", "day", "1day"):
        return "days"
    if s in ("1w", "w", "week", "1week"):
        return "weeks"
    if s in ("1mo", "mon", "month", "1month", "1mth"):
        return "months"
    return "days"


def _to_upbit_kst_string(dt_utc: datetime) -> str:
    """Upbit 'to' 파라미터: 'YYYY-MM-DD HH:MM:SS'(KST)."""
    kst = timezone(timedelta(hours=9))
    return dt_utc.astimezone(kst).strftime("%Y-%m-%d %H:%M:%S")


def fetch_upbit_ohlcv(
    market: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    max_rows: int = 10_000,
    session: "requests.Session | None" = None,
) -> pd.DataFrame:
    """Upbit Public API(무인증). market 예: 'KRW-BTC'."""
    if session is None:
        if _requests is None:
            raise RuntimeError("requests가 필요합니다. `pip install requests`")
        s = _requests.Session()
    else:
        s = session

    path = _upbit_interval_path(interval)
    url = f"{_UPBIT_BASE}/{path}"

    dt_start = _parse_dt(start)
    dt_end = _parse_dt(end)

    to_cursor_utc: datetime | None = dt_end
    rows: list[dict] = []

    while True:
        params = {"market": market, "count": 200}
        if to_cursor_utc:
            params["to"] = _to_upbit_kst_string(to_cursor_utc)

        r = s.get(url, params=params, timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"Upbit API 오류: HTTP {r.status_code} - {r.text}")

        batch = r.json()
        if not batch:
            break

        rows.extend(batch)
        if len(rows) >= max_rows:
            break

        oldest_utc = pd.to_datetime(batch[-1]["candle_date_time_utc"], utc=True).to_pydatetime()
        to_cursor_utc = oldest_utc - timedelta(seconds=1)
        if dt_start and (oldest_utc <= dt_start):
            break

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["candle_date_time_utc"], utc=True)
    df = df.set_index(idx)

    rename_map = {
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].copy()

    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if dt_start:
        df = df[df.index >= _as_utc_ts(dt_start)]
    if dt_end:
        df = df[df.index <= _as_utc_ts(dt_end)]

    df = _flatten_and_dedupe_columns(df)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return _finalize(df)

# ---------- 고수준 라우터 ----------

@dataclass(frozen=True)
class CollectResult:
    source: str
    symbol: str
    interval: str
    start: str | None
    end: str | None
    dataframe: pd.DataFrame


def collect(
    source: str,
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    **kwargs,
) -> CollectResult:
    """상위 수집: source ∈ {'yahoo','upbit'}."""
    s = source.strip().lower()
    if s == "yahoo":
        df = fetch_yahoo_ohlcv(symbol=symbol, start=start, end=end, interval=interval)
    elif s == "upbit":
        df = fetch_upbit_ohlcv(market=symbol, start=start, end=end, interval=interval, **kwargs)
    else:
        raise ValueError("source는 'yahoo' 또는 'upbit'만 지원합니다.")

    df.index = _to_utc_index(df.index)
    df = _finalize(df)
    return CollectResult(source=s, symbol=symbol, interval=interval, start=start, end=end, dataframe=df)

# ---------- CLI ----------

try:
    from requests import RequestException as _ReqExc  # type: ignore
except Exception:
    class _ReqExc(Exception):
        pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OHLCV 수집 (Yahoo/Upbit)")
    parser.add_argument("source", type=str, help="yahoo | upbit")
    parser.add_argument("symbol", type=str, help="예: AAPL / 005930.KS / KRW-BTC")
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD 또는 ISO8601 (UTC)")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD 또는 ISO8601 (UTC)")
    parser.add_argument("--interval", type=str, default="1d", help="yahoo/upbit 인터벌 문자열")
    parser.add_argument("--out-csv", type=str, default=None, help="CSV 저장 경로(선택)")
    args = parser.parse_args()

    try:
        result = collect(
            source=args.source,
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            interval=args.interval,
        )
        df = result.dataframe
        if args.out_csv:
            Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.out_csv, index=True)
            print(f"[OK] rows={len(df):,} → CSV: {args.out_csv}")
        else:
            print(f"[SOURCE] {result.source}  [SYMBOL] {result.symbol}  [INTERVAL] {result.interval}")
            print(f"[PERIOD] start={result.start} end={result.end}  [ROWS] {len(df):,}")
            print(df.head().to_string())
    except (_ReqExc, ValueError, RuntimeError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        print(
            "예시:\n"
            "  python collect.py yahoo AAPL --start 2020-01-01 --end 2025-01-01 --interval 1d\n"
            "  python collect.py upbit KRW-BTC --start 2023-01-01 --end 2025-01-01 --interval 1d",
            file=sys.stderr,
        )
        sys.exit(1)

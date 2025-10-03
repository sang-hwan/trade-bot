# data/collect.py
"""
OHLCV 수집 → UTC DatetimeIndex → 표준 컬럼화(open, high, low, close[, volume, AdjClose])

공개 API
- fetch_yahoo_ohlcv(symbol, start=None, end=None, interval='1d') -> pd.DataFrame
- fetch_upbit_ohlcv(market, start=None, end=None, interval='1d', max_rows=200, session=None) -> pd.DataFrame
- collect(source, symbol, start=None, end=None, interval='1d', *, base_currency=None, calendar_id=None,
          price_currency=None) -> CollectResult

출력 계약
- 인덱스: DatetimeIndex(tz='UTC'), 오름차순, 중복 제거
- 필수 컬럼: open, high, low, close (float)
- 선택 컬럼: volume (float), AdjClose (주식)
- 메타: meta={"price_currency": ...} (상위 qv_meta 연계용)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import time  # API rate-limit 보호에 사용
import pandas as pd

__all__ = [
    "CollectResult",
    "fetch_yahoo_ohlcv",
    "fetch_upbit_ohlcv",
    "collect",
]


# ───────────────────────────────
# 공용 유틸
# ───────────────────────────────

def _as_utc_ts(x: str | datetime | pd.Timestamp) -> pd.Timestamp:
    """문자열/Datetime을 tz-aware UTC Timestamp로 변환."""
    ts = pd.Timestamp(x)
    return ts.tz_localize(timezone.utc) if ts.tzinfo is None else ts.tz_convert(timezone.utc)


def _to_utc_index(idx: Iterable) -> pd.DatetimeIndex:
    """임의 인덱스를 UTC로 통일하고 정렬·중복 제거."""
    di = pd.DatetimeIndex(idx)
    di = di.tz_localize(timezone.utc) if di.tz is None else di.tz_convert(timezone.utc)
    return pd.DatetimeIndex(di.sort_values().unique())


def _enforce_ohlc_types(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV 수치형 강제."""
    for c in ("open", "high", "low", "close", "volume", "AdjClose"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _drop_dup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """중복 컬럼 제거(동일 이름이 여러 개면 마지막 것 유지)."""
    if not isinstance(df.columns, pd.Index):
        return df
    return df.loc[:, ~df.columns.duplicated(keep="last")]


def _drop_dupes_sort(df: pd.DataFrame) -> pd.DataFrame:
    """중복 제거 후 시계열 정렬."""
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def _map_yahoo_interval(iv: str) -> str:
    """사용자 인터벌 → yfinance 인터벌."""
    m = {"1d": "1d", "1w": "1wk", "1wk": "1wk", "1m": "1mo", "1mo": "1mo"}
    return m.get(iv.lower(), iv)


def _map_upbit_interval(iv: str) -> str:
    """사용자 인터벌 → Upbit 일/주/월 엔드포인트 키."""
    ivu = iv.lower()
    if ivu in ("1d", "day", "1day"):
        return "days"
    if ivu in ("1w", "week", "1week"):
        return "weeks"
    if ivu in ("1m", "1mo", "month", "1month"):
        return "months"
    raise ValueError(f"Unsupported Upbit interval: {iv}")


def _infer_price_currency(source: str, symbol: str) -> str:
    """
    로컬(가격) 통화 추정:
    - upbit: 'KRW-BTC' → 'KRW', 'USDT-XXX' → 'USDT', 'BTC-XXX' → 'BTC'
    - yahoo: 티커 접미사 휴리스틱(기본 'USD')
    """
    s = source.lower()
    sym = symbol.upper()
    if s == "upbit":
        return sym.split("-", 1)[0] if "-" in sym else "KRW"
    suffix_map = {
        "KS": "KRW", "KQ": "KRW", "T": "JPY",
        "TO": "CAD", "V": "CAD", "L": "GBX",
        "HK": "HKD", "SS": "CNY", "SZ": "CNY",
        "SI": "SGD", "AX": "AUD", "NS": "INR",
        "BO": "INR", "SA": "BRL", "MX": "MXN",
        "MI": "EUR", "PA": "EUR", "AS": "EUR",
        "DE": "EUR", "BE": "EUR", "SW": "CHF",
    }
    if "." in sym:
        return suffix_map.get(sym.rsplit(".", 1)[-1], "USD")
    return "USD"


# ───────────────────────────────
# 원천별 수집
# ───────────────────────────────

def fetch_yahoo_ohlcv(
    symbol: str,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Yahoo Finance OHLCV(+AdjClose) → 표준 스키마."""
    try:
        import yfinance as yf  # 서드파티
    except ImportError as e:
        raise RuntimeError("yfinance가 필요합니다. pip install yfinance") from e

    yf_iv = _map_yahoo_interval(interval)
    kwargs: dict[str, Any] = {}
    if start:
        kwargs["start"] = _as_utc_ts(start).to_pydatetime()
    if end:
        # yfinance는 end 비포함 → 다음날 00:00으로 보정
        kwargs["end"] = (_as_utc_ts(end) + pd.Timedelta(days=1)).to_pydatetime()

    dl = yf.download(symbol, interval=yf_iv, auto_adjust=False, progress=False, **kwargs)
    if not isinstance(dl, pd.DataFrame) or dl.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    dl = dl.rename(
        columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "AdjClose", "Volume": "volume",
        }
    )
    dl = _drop_dup_columns(dl)
    dl.index = _to_utc_index(dl.index)

    if start:
        dl = dl[dl.index >= _as_utc_ts(start)]
    if end:
        dl = dl[dl.index <= _as_utc_ts(end)]

    dl = _enforce_ohlc_types(dl)
    dl = _drop_dupes_sort(dl)

    for c in ("open", "high", "low", "close"):
        if c not in dl.columns:
            dl[c] = pd.NA
    if "volume" not in dl.columns:
        dl["volume"] = pd.NA

    return dl[["open", "high", "low", "close", "volume"]].assign(
        AdjClose=dl.get("AdjClose", pd.NA)
    )


def fetch_upbit_ohlcv(
    market: str,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    interval: str = "1d",
    *,
    max_rows: int = 200,
    session=None,
) -> pd.DataFrame:
    """Upbit Public API(무인증): 일/주/월 봉 → 표준 스키마."""
    import requests

    endpoint_kind = _map_upbit_interval(interval)
    base_url = f"https://api.upbit.com/v1/candles/{endpoint_kind}"
    sess = session or requests.Session()

    dt_start = _as_utc_ts(start) if start else None
    dt_end = _as_utc_ts(end) if end else None
    to_cursor = dt_end or pd.Timestamp.now(tz=timezone.utc)
    frames: list[pd.DataFrame] = []

    while True:
        params = {"market": market, "count": max_rows}
        # Upbit 'to'는 KST 기준 문자열
        to_kst = to_cursor.tz_convert("Asia/Seoul")
        params["to"] = to_kst.strftime("%Y-%m-%d %H:%M:%S")

        r = sess.get(base_url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        df = pd.DataFrame(data)
        df["ts"] = pd.to_datetime(df["candle_date_time_utc"], utc=True)
        df = df.set_index("ts")[["opening_price", "high_price", "low_price", "trade_price", "candle_acc_trade_volume"]]
        df = df.rename(
            columns={
                "opening_price": "open",
                "high_price": "high",
                "low_price": "low",
                "trade_price": "close",
                "candle_acc_trade_volume": "volume",
            }
        )
        df = _drop_dup_columns(df)
        frames.append(df)

        last_ts = df.index.min()  # 최신순 반환 → min이 과거
        if dt_start and last_ts <= dt_start:
            break
        to_cursor = (last_ts - pd.Timedelta(seconds=1)).tz_convert(timezone.utc)
        time.sleep(0.05)  # rate-limit 보호

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    dl = pd.concat(frames, axis=0)
    dl.index = _to_utc_index(dl.index)

    if start:
        dl = dl[dl.index >= _as_utc_ts(start)]
    if end:
        dl = dl[dl.index <= _as_utc_ts(end)]

    dl = _enforce_ohlc_types(dl)
    dl = _drop_dupes_sort(dl)
    return dl[["open", "high", "low", "close", "volume"]]


# ───────────────────────────────
# 수집 파사드
# ───────────────────────────────

@dataclass(frozen=True)
class CollectResult:
    """상위 게이트/스냅샷 연계용 컨테이너."""
    dataframe: pd.DataFrame
    base_currency: str | None = None
    calendar_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)  # 예: {"price_currency": "USD"}


def collect(
    source: str,
    symbol: str,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    interval: str = "1d",
    *,
    base_currency: str | None = None,
    calendar_id: str | None = None,
    price_currency: str | None = None,
) -> CollectResult:
    """
    원천별 수집 후 UTC·표준화 완료된 DataFrame과 메타를 반환.
    meta['price_currency']는 qv_meta(...)의 FX 필요 판단에 사용 가능.
    """
    src = source.lower()
    if src == "yahoo":
        df = fetch_yahoo_ohlcv(symbol, start=start, end=end, interval=interval)
        pc = price_currency or _infer_price_currency("yahoo", symbol)
        return CollectResult(df, base_currency=base_currency, calendar_id=calendar_id, meta={"price_currency": pc})

    if src == "upbit":
        df = fetch_upbit_ohlcv(symbol, start=start, end=end, interval=interval)
        cal = calendar_id or "24x7"
        pc = price_currency or _infer_price_currency("upbit", symbol)
        return CollectResult(df, base_currency=base_currency, calendar_id=cal, meta={"price_currency": pc})

    raise ValueError(f"Unsupported source: {source}")


# ───────────────────────────────
# CLI (선택)
# ───────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Collect OHLCV data (UTC standardized)")
    p.add_argument("--source", required=True, help="yahoo | upbit")
    p.add_argument("--symbol", required=True, help="예: AAPL / 005930.KS / KRW-BTC")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--interval", default="1d")
    p.add_argument("--base-currency", dest="base_currency", default=None)
    p.add_argument("--calendar-id", dest="calendar_id", default=None)
    p.add_argument("--price-currency", dest="price_currency", default=None)
    p.add_argument("--out-csv", default=None)
    args = p.parse_args()

    result = collect(
        source=args.source,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        interval=args.interval,
        base_currency=args.base_currency,
        calendar_id=args.calendar_id,
        price_currency=args.price_currency,
    )

    print(
        f"[collect] rows={len(result.dataframe)}, "
        f"cols={list(result.dataframe.columns)}, "
        f"tz={getattr(result.dataframe.index, 'tz', None)}"
    )
    print(f"[meta] base_currency={result.base_currency} calendar_id={result.calendar_id} meta={result.meta}")

    if args.out_csv:
        result.dataframe.to_csv(args.out_csv, index=True)
        print(f"[saved] {args.out_csv}")

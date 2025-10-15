# data/collect.py
"""
OHLCV 수집 → UTC DatetimeIndex → 표준 컬럼(open, high, low, close[, volume, AdjClose])

공개 API
- fetch_yahoo_ohlcv(symbol, start=None, end=None, interval='1d') -> pd.DataFrame
- fetch_upbit_ohlcv(market, start=None, end=None, interval='1d', max_rows=200, session=None) -> pd.DataFrame
- collect(source, symbol, start=None, end=None, interval='1d', *, base_currency=None, calendar_id=None,
          price_currency=None) -> CollectResult

(추가) 유니버스/마스터 수집 API
- list_upbit_markets(quote="KRW") -> pd.DataFrame            # Upbit /v1/market/all
- fetch_upbit_tickers(markets: list[str]) -> pd.DataFrame    # Upbit /v1/ticker (acc_trade_price_24h 포함)
- list_us_equities_from_nasdaq_dir() -> pd.DataFrame         # nasdaqlisted.txt + otherlisted.txt 병합(Test Issue 제외)

규약
- 인덱스: DatetimeIndex(tz='UTC'), 오름차순, 중복 제거
- 필수 컬럼: open, high, low, close (float)
- 선택 컬럼: volume (float), AdjClose (주식)
- 메타: meta={"price_currency": ...}

예외 처리 근거
- HTTP/네트워크/JSON 오류는 RuntimeError로 승격해 호출자가 원인 메시지를 확인 가능하도록 함.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional, Tuple

import csv
import json
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError

import pandas as pd

__all__ = [
    "CollectResult",
    "fetch_yahoo_ohlcv",
    "fetch_upbit_ohlcv",
    "collect",
    "list_upbit_markets",
    "fetch_upbit_tickers",
    "list_us_equities_from_nasdaq_dir",
]


# ── 공용 유틸 ─────────────────────────────────────────────────────────────────

def _as_utc_ts(x: str | datetime | pd.Timestamp) -> pd.Timestamp:
    """문자열/Datetime → tz-aware UTC Timestamp."""
    ts = pd.Timestamp(x)
    return ts.tz_localize(timezone.utc) if ts.tzinfo is None else ts.tz_convert(timezone.utc)


def _to_utc_index(idx: Iterable) -> pd.DatetimeIndex:
    """인덱스를 UTC로 통일하고 정렬·중복 제거."""
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
    """동일 컬럼명 중복 제거(마지막 것 유지)."""
    if not isinstance(df.columns, pd.Index):
        return df
    return df.loc[:, ~df.columns.duplicated(keep="last")]


def _drop_dupes_sort(df: pd.DataFrame) -> pd.DataFrame:
    """중복 인덱스 제거 후 정렬."""
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def _map_yahoo_interval(iv: str) -> str:
    """사용자 인터벌 → yfinance 인터벌."""
    m = {"1d": "1d", "1w": "1wk", "1wk": "1wk", "1m": "1mo", "1mo": "1mo"}
    return m.get(iv.lower(), iv)


def _map_upbit_interval(iv: str) -> str:
    """사용자 인터벌 → Upbit 엔드포인트 키."""
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


def _http_get_json(url: str, *, timeout: int = 10) -> Any:
    """표준 라이브러리 urllib 기반 JSON GET."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore") if e.fp else ""
        raise RuntimeError(f"HTTP {e.code} {detail} (url={url})") from None
    except URLError as e:
        raise RuntimeError(f"연결 오류: {e.reason} (url={url})") from None
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON 디코드 오류: {e} (url={url})") from None


def _http_get_text(url: str, *, timeout: int = 15, encoding: str = "utf-8") -> str:
    """표준 라이브러리 urllib 기반 텍스트 GET."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                return raw.decode("latin-1")
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore") if e.fp else ""
        raise RuntimeError(f"HTTP {e.code} {detail} (url={url})") from None
    except URLError as e:
        raise RuntimeError(f"연결 오류: {e.reason} (url={url})") from None


# ── 원천별 OHLCV 수집 ─────────────────────────────────────────────────────────

def fetch_yahoo_ohlcv(
    symbol: str,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Yahoo Finance OHLCV(+AdjClose) → 표준 스키마."""
    try:
        import yfinance as yf  # 프로젝트 종속성
    except ImportError as e:
        raise RuntimeError("yfinance가 필요합니다. pip install yfinance") from e

    yf_iv = _map_yahoo_interval(interval)
    kwargs: dict[str, Any] = {}
    if start:
        kwargs["start"] = _as_utc_ts(start).to_pydatetime()
    if end:
        kwargs["end"] = (_as_utc_ts(end) + pd.Timedelta(days=1)).to_pydatetime()  # yfinance end는 비포함

    dl = yf.download(
        symbol,
        interval=yf_iv,
        auto_adjust=False,
        progress=False,
        group_by="column",
        **kwargs,
    )
    if not isinstance(dl, pd.DataFrame) or dl.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if isinstance(dl.columns, pd.MultiIndex):
        lvl0 = {str(x).lower() for x in dl.columns.get_level_values(0)}
        ohlcv = {"open", "high", "low", "close", "adj close", "volume"}
        dl = dl.droplevel(1, axis=1) if len(ohlcv & lvl0) >= 3 else dl.droplevel(0, axis=1)

    dl = dl.rename(
        columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "AdjClose", "Volume": "volume",
        }
    )
    if isinstance(dl.columns, pd.MultiIndex):
        raise ValueError("Yahoo 컬럼이 여전히 MultiIndex입니다. 단일 티커로 요청하세요.")

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
    session=None,  # 호환 목적(미사용)
) -> pd.DataFrame:
    """Upbit Public API(무인증): 일/주/월 봉 → 표준 스키마. 'to'는 KST 문자열, 최신→과거 페이징."""
    endpoint_kind = _map_upbit_interval(interval)
    base_url = f"https://api.upbit.com/v1/candles/{endpoint_kind}"

    dt_start = _as_utc_ts(start) if start else None
    dt_end = _as_utc_ts(end) if end else None
    to_cursor = dt_end or pd.Timestamp.now(tz=timezone.utc)
    frames: list[pd.DataFrame] = []

    while True:
        params = {"market": market, "count": max_rows}
        to_kst = to_cursor.tz_convert("Asia/Seoul")
        params["to"] = to_kst.strftime("%Y-%m-%d %H:%M:%S")

        url = base_url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore") if e.fp else ""
            raise RuntimeError(f"Upbit HTTP {e.code} {detail}") from None
        except URLError as e:
            raise RuntimeError(f"Upbit 연결 오류: {e.reason}") from None
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Upbit JSON 오류: {e}") from None

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

        last_ts = df.index.min()  # 최신→과거 반환이므로 min이 더 과거
        if dt_start and last_ts <= dt_start:
            break
        to_cursor = (last_ts - pd.Timedelta(seconds=1)).tz_convert(timezone.utc)
        time.sleep(0.05)  # 레이트리밋 보호

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


# ── (추가) 유니버스/마스터 수집 ────────────────────────────────────────────────

def list_upbit_markets(quote: str = "KRW") -> pd.DataFrame:
    """
    Upbit 마켓 전수 목록(/v1/market/all?isDetails=true) → KRW-* 필터.
    반환 컬럼: 가능한 경우 [market, korean_name, english_name, market_warning, market_event]
    """
    url = "https://api.upbit.com/v1/market/all?isDetails=true"
    data = _http_get_json(url)
    if not isinstance(data, list):
        return pd.DataFrame(columns=["market", "korean_name", "english_name"])

    df = pd.DataFrame(data)
    keep_cols = [c for c in ("market", "korean_name", "english_name", "market_warning", "market_event") if c in df.columns]
    if not keep_cols:
        keep_cols = ["market"]
    df = df[keep_cols]
    df = df[df["market"].str.startswith(f"{quote}-", na=False)]
    return df.sort_values("market").reset_index(drop=True)


def _chunk(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i: i + n]


def fetch_upbit_tickers(markets: List[str]) -> pd.DataFrame:
    """
    Upbit Ticker 벌크 조회(/v1/ticker?markets=...).
    반환 컬럼(가능 시): market, trade_price, high_price, low_price, acc_trade_price_24h, acc_trade_volume_24h,
                       signed_change_price, signed_change_rate, prev_closing_price
    """
    if not markets:
        return pd.DataFrame(columns=[
            "market", "trade_price", "high_price", "low_price",
            "acc_trade_price_24h", "acc_trade_volume_24h", "signed_change_rate"
        ])

    rows: list[pd.DataFrame] = []
    for group in _chunk([m.strip().upper() for m in markets if m and isinstance(m, str)], 100):
        q = ",".join(group)
        url = "https://api.upbit.com/v1/ticker?" + urllib.parse.urlencode({"markets": q})
        data = _http_get_json(url)
        if not data:
            continue
        df = pd.DataFrame(data)
        keep = [c for c in (
            "market", "trade_price", "high_price", "low_price",
            "acc_trade_price_24h", "acc_trade_volume_24h",
            "signed_change_price", "signed_change_rate", "prev_closing_price"
        ) if c in df.columns]
        df = df[keep]
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=[
            "market", "trade_price", "high_price", "low_price",
            "acc_trade_price_24h", "acc_trade_volume_24h", "signed_change_rate"
        ])

    out = pd.concat(rows, axis=0, ignore_index=True)
    num_cols = [c for c in (
        "trade_price", "high_price", "low_price",
        "acc_trade_price_24h", "acc_trade_volume_24h",
        "signed_change_price", "signed_change_rate", "prev_closing_price"
    ) if c in out.columns]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _parse_symbol_dir_text(text: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    NASDAQ SymbolDirectory 텍스트(| 구분) → DataFrame, 'File Creation Time' 문자열(있을 경우).
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    file_ctime: Optional[str] = None
    if lines and lines[-1].lower().startswith("file creation time"):
        file_ctime = lines[-1].split(":", 1)[-1].strip()
        lines = lines[:-1]
    if not lines:
        return pd.DataFrame(), file_ctime

    reader = csv.DictReader(lines, delimiter="|")
    rows = list(reader)
    if not rows:
        return pd.DataFrame(), file_ctime
    return pd.DataFrame(rows), file_ctime


def list_us_equities_from_nasdaq_dir(
    url_nasdaq: str = "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt",
    url_other: str = "https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt",
) -> pd.DataFrame:
    """
    NASDAQ 공식 Symbol Directory 병합: nasdaqlisted.txt + otherlisted.txt (Test Issue 제외).
    반환 컬럼: symbol, security_name, exchange, is_etf, round_lot_size, source_file, file_creation_time
    """
    txt1 = _http_get_text(url_nasdaq)
    txt2 = _http_get_text(url_other)

    df1, ctime1 = _parse_symbol_dir_text(txt1)
    df2, ctime2 = _parse_symbol_dir_text(txt2)

    std_rows: list[pd.DataFrame] = []

    if not df1.empty:
        # Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
        df1 = df1.rename(columns={
            "Symbol": "symbol",
            "Security Name": "security_name",
            "Round Lot Size": "round_lot_size",
            "ETF": "ETF",
            "Test Issue": "Test Issue",
        })
        df1["exchange"] = "NASDAQ"
        if "round_lot_size" in df1.columns:
            df1["round_lot_size"] = pd.to_numeric(df1["round_lot_size"], errors="coerce")
        df1["is_etf"] = df1.get("ETF", "").astype(str).upper().eq("Y")
        df1["test_issue"] = df1.get("Test Issue", "").astype(str).upper().eq("Y")
        df1["source_file"] = "nasdaqlisted.txt"
        df1["file_creation_time"] = ctime1
        std_rows.append(df1[["symbol", "security_name", "exchange", "is_etf", "round_lot_size", "test_issue", "source_file", "file_creation_time"]])

    if not df2.empty:
        # Symbol|Exchange|Security Name|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
        df2 = df2.rename(columns={
            "Symbol": "symbol",
            "Security Name": "security_name",
            "Exchange": "exchange",
            "Round Lot Size": "round_lot_size",
            "ETF": "ETF",
            "Test Issue": "Test Issue",
        })
        exch_map = {"N": "NYSE", "A": "AMEX", "P": "ARCA", "Z": "BATS"}
        df2["exchange"] = df2["exchange"].map(lambda x: exch_map.get(str(x).upper(), str(x).upper()))
        if "round_lot_size" in df2.columns:
            df2["round_lot_size"] = pd.to_numeric(df2["round_lot_size"], errors="coerce")
        df2["is_etf"] = df2.get("ETF", "").astype(str).upper().eq("Y")
        df2["test_issue"] = df2.get("Test Issue", "").astype(str).upper().eq("Y")
        df2["source_file"] = "otherlisted.txt"
        df2["file_creation_time"] = ctime2
        std_rows.append(df2[["symbol", "security_name", "exchange", "is_etf", "round_lot_size", "test_issue", "source_file", "file_creation_time"]])

    if not std_rows:
        return pd.DataFrame(columns=["symbol", "security_name", "exchange", "is_etf", "round_lot_size", "source_file", "file_creation_time"])

    merged = pd.concat(std_rows, axis=0, ignore_index=True)
    if "test_issue" in merged.columns:
        merged = merged[~merged["test_issue"]].drop(columns=["test_issue"])
    merged = merged.sort_values(["symbol", "source_file"], ascending=[True, True])
    merged = merged.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    return merged


# ── 수집 파사드 ────────────────────────────────────────────────────────────────

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
    """원천별 수집 후 UTC·표준화 DataFrame과 메타를 반환."""
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


# ── CLI ───────────────────────────────────────────────────────────────────────

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

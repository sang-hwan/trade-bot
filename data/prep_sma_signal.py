"""
조정 종가(Adj Close)로 보정한 OHLC를 기반으로 SMA10/50 시그널을 룩어헤드 없이 산출하고 Parquet로 저장하는 전처리 스크립트

이 파일의 목적:
- OHLCV 시계열을 UTC·단조 인덱스로 정규화하고, 품질 게이트와 완전 조정(Adj Close/Close)을 거쳐 *_adj 컬럼을 생성한다.
- close_adj로 SMA10/50을 계산하고 t−1의 확정값으로 시그널(diff>ε→1, 그 외 0)을 산출한 뒤, 결과를 Parquet과 실행 요약(JSON)으로 고정한다.

사용되는 변수와 함수 목록:
- 변수
  - ticker: str|None — 데이터 소스 심볼(Yahoo 사용 시)
  - start: str|None — 수집 시작일(YYYY-MM-DD)
  - end: str|None — 수집 종료일(YYYY-MM-DD)
  - interval: str = "1d" — 봉 간격
  - input: str|None — 입력 파일 경로(.parquet | .csv)
  - out: str|None — 출력 경로(파일 또는 디렉터리)
  - epsilon: float = 0.0 — 타이브레이크 임계(ε ≥ 0)
  - columns: open, high, low, close, adj_close → open_adj, high_adj, low_adj, close_adj, sma10, sma50, signal(0/1)
- 함수
  - _read_input(path: str)
    - 역할: Parquet/CSV 입력 로드
    - 입력값: path: str - 파일 경로
    - 반환값: pandas.DataFrame - 원본 시계열
  - _fetch_yahoo(ticker: str, start: str|None, end: str|None, interval: str)
    - 역할: Yahoo Finance에서 OHLCV 수집(auto_adjust=False)
    - 입력값: ticker: str, start: str|None, end: str|None, interval: str
    - 반환값: pandas.DataFrame - 시계열
  - _to_utc_index(df: pandas.DataFrame)
    - 역할: timestamp 인덱스화 및 UTC 변환
    - 입력값: df: DataFrame
    - 반환값: pandas.DataFrame - tz-aware UTC 인덱스
  - _normalize_columns(df: pandas.DataFrame)
    - 역할: 컬럼명 정규화 및 필수 컬럼 검증, adj_close 없을 시 close로 대체
    - 입력값: df: DataFrame
    - 반환값: pandas.DataFrame
  - _quality_gate(df: pandas.DataFrame)
    - 역할: 정렬·중복 제거·NaN 제거 및 OHLC 불변식(0<low≤min(open,close)≤max(open,close)≤high) 필터
    - 입력값: df: DataFrame
    - 반환값: pandas.DataFrame - 유효 행만 유지
  - _adjust_ohlc(df: pandas.DataFrame)
    - 역할: a_t=adj_close/close로 O/H/L/C 완전 조정하여 *_adj 생성
    - 입력값: df: DataFrame
    - 반환값: pandas.DataFrame - open_adj/high_adj/low_adj/close_adj 포함
  - _sma(series: pandas.Series, n: int)
    - 역할: 고정창 단순이동평균 계산(rolling, min_periods=n)
    - 입력값: series: Series, n: int
    - 반환값: pandas.Series - SMA_n
  - _compute_signal(df: pandas.DataFrame, epsilon: float)
    - 역할: close_adj로 SMA10/50 산출, diff_prev=(SMA10−SMA50).shift(1)>ε 판정으로 signal 생성
    - 입력값: df: DataFrame, epsilon: float
    - 반환값: pandas.DataFrame - sma10/sma50/signal 추가
  - _first_valid_indices(df: pandas.DataFrame)
    - 역할: sma10/sma50 최초 생성 시점과 첫 유효 시그널 시점 추출
    - 입력값: df: DataFrame
    - 반환값: tuple - (first_sma10, first_sma50, first_valid_signal)
  - _default_outpath(out: str|None, ticker: str|None)
    - 역할: 출력 파일 경로 기본값 생성(YYYYMMDD 접미)
    - 입력값: out: str|None, ticker: str|None
    - 반환값: str - 경로
  - _config_id(cfg: dict)
    - 역할: 설정 JSON을 md5로 해싱해 식별자 생성
    - 입력값: cfg: dict
    - 반환값: str - 해시 앞 10자
  - main()
    - 역할: 인자 파싱→로드→정규화→품질 게이트→조정→SMA/시그널→저장→요약 출력의 전체 파이프라인 실행
    - 입력값: CLI 인자(--input|--ticker, --start, --end, --interval, --out, --epsilon)
    - 반환값: None

파일의 흐름(→ / ->):
- 입력 로드(파일 또는 Yahoo) -> UTC 인덱스화/컬럼 정규화 -> 품질 게이트(정렬·중복·NaN·OHLC 불변식)
- 조정 OHLC 생성(*_adj) -> SMA10/50 계산(close_adj) -> t−1 기준 diff_prev로 시그널 산출(signal∈{0,1})
- 최초 유효 시점 추출 -> 출력 경로/설정 해시 결정 -> Parquet 저장 -> 실행 요약(JSON) 출력
"""

import argparse
import os
import json
import hashlib
from datetime import datetime, timezone
import pandas as pd

def _read_input(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    raise ValueError("unsupported input format")

def _fetch_yahoo(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    import yfinance as yf
    data = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)
    data = data.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
    data.index.name = "timestamp"
    return data

def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=False, errors="coerce")
    df.index = (df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC"))
    return df

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    required = ["open","high","low","close"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    for c in ["open","high","low","close","adj_close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _quality_gate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    base = ["open","high","low","close","adj_close"]
    df = df.dropna(subset=base)
    mask = (df[["open","high","low","close"]]>0).all(axis=1)
    oc_min = df[["open","close"]].min(axis=1)
    oc_max = df[["open","close"]].max(axis=1)
    mask &= (df["low"]<=oc_min) & (oc_max<=df["high"])
    return df[mask]

def _adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    factor = df["adj_close"] / df["close"]
    df["open_adj"]  = df["open"]  * factor
    df["high_adj"]  = df["high"]  * factor
    df["low_adj"]   = df["low"]   * factor
    df["close_adj"] = df["close"] * factor
    return df

def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n, min_periods=n).mean()

def _compute_signal(df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    df = df.copy()
    df["sma10"] = _sma(df["close_adj"], 10)
    df["sma50"] = _sma(df["close_adj"], 50)
    diff_prev = df["sma10"].shift(1) - df["sma50"].shift(1)
    df["signal"] = (diff_prev > epsilon).astype("int8").where(diff_prev.notna(), 0)
    return df

def _first_valid_indices(df: pd.DataFrame):
    i_sma10 = df["sma10"].first_valid_index()
    i_sma50 = df["sma50"].first_valid_index()
    mask = df["sma10"].shift(1).notna() & df["sma50"].shift(1).notna()
    i_sig = df.index[mask][0] if mask.any() else None
    return i_sma10, i_sma50, i_sig

def _default_outpath(out: str, ticker: str|None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    base = f"sma1050_signal_{ticker.lower() if ticker else 'data'}_{ts}.parquet"
    if not out:
        return base
    if os.path.isdir(out):
        return os.path.join(out, base)
    return out

def _config_id(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:10]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default=None)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--epsilon", type=float, default=0.0)
    args = p.parse_args()

    if not args.input and not args.ticker:
        raise SystemExit("require --input or --ticker")

    df = _read_input(args.input) if args.input else _fetch_yahoo(args.ticker, args.start, args.end, args.interval)
    n0 = len(df)
    df = _to_utc_index(df)
    df = _normalize_columns(df)
    df = _quality_gate(df)
    df = _adjust_ohlc(df)
    if len(df) < 51:
        raise SystemExit("need at least 51 rows after cleaning")
    df = _compute_signal(df, args.epsilon)
    i_sma10, i_sma50, i_sig = _first_valid_indices(df)

    cfg = {
        "source": "input" if args.input else "yahoo",
        "ticker": args.ticker,
        "epsilon": args.epsilon,
        "interval": args.interval,
        "start": args.start,
        "end": args.end,
    }
    cid = _config_id(cfg)

    out_path = _default_outpath(args.out, args.ticker)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    df.to_parquet(out_path, index=True)

    print(json.dumps({
        "rows_in": n0,
        "rows_out": len(df),
        "first_sma10": str(i_sma10) if i_sma10 else None,
        "first_sma50": str(i_sma50) if i_sma50 else None,
        "first_valid_signal": str(i_sig) if i_sig else None,
        "epsilon": args.epsilon,
        "out": out_path,
        "config_id": cid
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()

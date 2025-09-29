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
    df = df[~df.index.duplicated(keep="last")]
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

"""
이 파일의 목적
- config.json(1단계 산출물)을 읽고, 단일 소스(Yahoo)에서 OHLCV를 내려받아 UTC 기준으로 정렬/정합화한 뒤 저장한다.
- 출처·설정 일관성을 유지하여 이후 단계(전략/시뮬레이터)에서 재현 가능한 입력을 제공한다.

사용되는 변수와 함수 목록
> 사용되는 변수의 의미
- INTERVAL_MAP: dict[str,str] — 설정(timeframe)을 yfinance interval("1d","60m","15m","5m")로 매핑
- args: argparse.Namespace — 명령줄 인자 컨테이너(--config, --out)
- cfg_path: pathlib.Path — 설정 파일 경로
- cfg: dict — 설정 내용(JSON 디코드 결과)
- tickers: list[str] — 심볼 목록
- start / end: str — 데이터 수집 시작/종료일(YYYY-MM-DD)
- timeframe: str — 캔들 주기("1D","1H","15m","5m")
- interval: str — yfinance interval("1d","60m","15m","5m")
- df: pandas.DataFrame — 다운로드 후 정합화한 시계열 데이터
- outdir: pathlib.Path — 결과 저장 폴더
- out_path: pathlib.Path — 결과 파일 경로(.parquet 우선, 실패 시 .csv)

> 사용되는 함수의 의미
- load_ohlcv(tickers, start, end, interval): Yahoo에서 OHLCV를 내려받고 UTC 변환·중복 제거·정렬·컬럼 표준화까지 수행
- main(): 설정을 읽고(load_ohlcv 호출) 결과를 디스크에 저장한 뒤 요약을 출력

> 파일의 화살표 프로세스
config.json 로드 → timeframe 유효성 확인 및 interval 매핑 → load_ohlcv로 다운로드/정합화(UTC, 중복 제거, 정렬)
→ {out}/{timeframe}/ohlcv_{timeframe}_{config_id}.parquet 저장(실패 시 .csv) → 콘솔 요약 출력

> 사용되는 함수의 입력값 목록과 그 의미
- load_ohlcv(tickers: list[str], start: str, end: str, interval: str)
  • tickers: 심볼 리스트(예: ["AAPL","MSFT"])
  • start/end: 기간 경계(YYYY-MM-DD)
  • interval: yfinance 인터벌("1d","60m","15m","5m")
- main()
  • CLI 인자: --config(설정 파일 경로, 기본 data/config_snapshots/config.json), --out(저장 폴더, 기본 data/ohlcv)

> 사용되는 함수의 출력값 목록과 그 의미
- load_ohlcv → pandas.DataFrame: UTC 인덱스, 중복 제거·정렬 완료, 멀티티커는 평탄화된 컬럼명("TICKER_Field")
- main → None: 파일 생성의 부수효과(.parquet 또는 .csv)와 콘솔 요약 출력
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yfinance as yf

INTERVAL_MAP = {"1D": "1d", "1H": "60m", "15m": "15m", "5m": "5m"}

def load_ohlcv(tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        " ".join(tickers),
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
    )
    if df.empty:
        raise RuntimeError("데이터가 비어 있습니다.")
    idx = df.index
    if getattr(idx, "tz", None) is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df[~df.index.duplicated(keep="first")].sort_index().dropna(how="all")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["{}_{}".format(a, b) for a, b in df.columns]
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="data/config_snapshots/config.json")
    p.add_argument("--out", default="data/ohlcv")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    tickers = list(cfg["tickers"])
    start = cfg["resolved_start"]
    end = cfg["resolved_end"]
    timeframe = cfg["timeframe"]
    if timeframe not in INTERVAL_MAP:
        raise ValueError(f"timeframe은 {list(INTERVAL_MAP)} 중 하나여야 합니다: {timeframe}")
    interval = INTERVAL_MAP[timeframe]

    print(f"[INFO] source=yahoo tickers={tickers} start={start} end={end} interval={interval} config_id={cfg['config_id']}")

    df = load_ohlcv(tickers, start, end, interval)

    outdir = Path(args.out) / timeframe
    outdir.mkdir(parents=True, exist_ok=True)

    out_path = outdir / f"ohlcv_{timeframe}_{cfg['config_id']}.parquet"
    try:
        df.to_parquet(out_path)
    except Exception:
        out_path = outdir / f"ohlcv_{timeframe}_{cfg['config_id']}.csv"
        df.to_csv(out_path, date_format="%Y-%m-%d %H:%M:%S%z")

    print(f"[DONE] saved={out_path} rows={len(df):,} cols={df.shape[1]} from={df.index.min()} to={df.index.max()}")

if __name__ == "__main__":
    main()

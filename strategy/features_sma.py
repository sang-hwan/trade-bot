"""
이 파일의 목적
- 클린된 OHLCV 시계열(Parquet/CSV)에 **SMA(10), SMA(50)** 피처만 추가하여 초보자용 최소 전략의 입력을 만든다. 입력 파일 형식을 보존하여 저장한다.

사용되는 변수와 함수 목록
> 사용되는 변수의 의미
- args: argparse.Namespace — CLI 인자(--in, --out)
- in_path: pathlib.Path — 입력 파일 경로(.parquet | .csv)
- out_path: pathlib.Path | None — 출력 파일 경로(미지정 시 '_feat' 접미사 자동 결정)
- df: pandas.DataFrame — DatetimeIndex(UTC) 기반 시계열 데이터
- windows: tuple[int, int] — SMA 윈도우 길이(기본 10, 50)

> 사용되는 함수의 의미
- load_frame(path): Parquet/CSV 자동 판독, DatetimeIndex를 UTC로 통일하고 정렬
- infer_close_columns(df): 'Close' 또는 '*_Close' 컬럼 집합을 추론
- add_sma_features(df, windows): Close 기준으로 SMA10/50 컬럼을 추가
- save_frame(df, in_path, out_path): 입력 형식(parquet/csv)을 보존하여 저장
- main(): CLI 엔트리포인트 — 전체 흐름 실행 및 요약 출력

> 파일의 화살표 프로세스(→ 흐름)
입력 로드 → Close 컬럼 추론 → SMA10/50 계산 → 저장 → 요약 출력

> 사용되는 함수의 입력값 목록과 그 의미
- CLI: --in (입력 경로), --out (출력 경로; 선택)
- load_frame(path: Path): 읽을 파일 경로
- infer_close_columns(df: DataFrame): Close 컬럼 탐지 대상 프레임
- add_sma_features(df: DataFrame, windows: tuple[int,int]): SMA 윈도우 길이
- save_frame(df: DataFrame, in_path: Path, out_path: Path | None): 저장 경로

> 사용되는 함수의 출력값 목록과 그 의미
- load_frame → pandas.DataFrame(UTC DatetimeIndex, 시간순 정렬)
- infer_close_columns → list[str](Close 컬럼명 목록)
- add_sma_features → pandas.DataFrame(SMA10/50 컬럼이 추가된 프레임)
- save_frame → pathlib.Path(실제로 저장된 파일 경로)
- main → None(콘솔에 추가 컬럼/저장 경로/행·열 요약 출력)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
import pandas as pd


def load_frame(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input must have a DatetimeIndex (first column as datetime).")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    return df


def infer_close_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str)]
    if "Close" in cols:
        return ["Close"]
    close_cols = [c for c in cols if c.lower().endswith("_close")]
    if close_cols:
        return close_cols
    raise ValueError("Close column not found. Expected 'Close' or '*_Close'.")


def add_sma_features(df: pd.DataFrame, windows: Tuple[int, int] = (10, 50)) -> pd.DataFrame:
    close_cols = infer_close_columns(df)
    for c in close_cols:
        prefix = "" if c == "Close" else c[: -len("_Close")]
        for w in windows:
            name = f"SMA{w}" if prefix == "" else f"{prefix}_SMA{w}"
            df[name] = df[c].rolling(window=w, min_periods=w).mean()
    return df


def save_frame(df: pd.DataFrame, in_path: Path, out_path: Path | None) -> Path:
    in_path = Path(in_path)
    if out_path is None:
        out_path = in_path.with_name(in_path.stem + "_feat" + in_path.suffix)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path)
    elif out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, date_format="%Y-%m-%d %H:%M:%S%z")
    else:
        if in_path.suffix.lower() == ".parquet":
            out_path = out_path.with_suffix(".parquet")
            df.to_parquet(out_path)
        else:
            out_path = out_path.with_suffix(".csv")
            df.to_csv(out_path, date_format="%Y-%m-%d %H:%M:%S%z")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Add minimal features: SMA(10), SMA(50)")
    ap.add_argument("--in", dest="inp", required=True, help="Input file (.parquet or .csv)")
    ap.add_argument("--out", dest="out", default=None, help="Output file (optional)")
    args = ap.parse_args()

    in_path = Path(args.inp)
    df = load_frame(in_path)
    df = add_sma_features(df, windows=(10, 50))
    out_path = save_frame(df, in_path, Path(args.out) if args.out else None)

    added = [c for c in df.columns if c.endswith("SMA10") or c.endswith("SMA50") or c in ("SMA10", "SMA50")]
    print(f"[FEATURE] added={added}")
    print(f"[DONE] saved={out_path} rows={len(df):,} cols={df.shape[1]}")


if __name__ == "__main__":
    main()

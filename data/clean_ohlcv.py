"""
이 파일의 목적
- 2단계(로드) 산출물(Parquet/CSV)을 읽어 타임존·순서 정합성을 보정하고 중복/빈행을 제거한다.
- 옵션에 따라 미국 정규장(09:30–16:00, America/New_York) 구간만 필터링해 신호 왜곡을 줄인다.
- 결과를 원본 파일과 같은 폴더에 *_clean 확장 이름으로 저장한다.

사용되는 변수와 함수 목록
> 사용되는 변수의 의미
- args: argparse.Namespace — CLI 인자 컨테이너
- cfg: dict — config.json 내용(파일 추론용)
- in_path/out_path: pathlib.Path — 입력/출력 파일 경로
- df: pandas.DataFrame — 시계열 데이터

> 사용되는 함수의 의미
- infer_input_path(cfg, base_dir): config_id와 timeframe으로 표준 입력 파일 경로 추론
- load_df(path): Parquet/CSV 자동 판별 로딩
- clean_df(df, session, exchange_tz, open_str, close_str, timeframe): UTC 정합화·중복 제거·정렬·세션 필터
- save_df(df, out_path): Parquet 우선 저장(실패 시 CSV 폴백)
- main(): 인자 파싱→입력 경로 확정→클린·정렬→저장 및 요약 출력

> 파일의 화살표 프로세스
config.json 로드(또는 --in 직접 지정) → 입력 파일 결정 → load_df
→ clean_df(UTC 통일, 중복 제거, 정렬, (옵션) 정규장 필터)
→ *_clean.parquet 저장(실패 시 *_clean.csv) → 콘솔 요약 출력

> 사용되는 함수의 입력값 목록과 그 의미
- --config: 설정 파일 경로(기본: data/config_snapshots/config.json)
- --in: 입력 파일 경로(미지정 시 config로 추론: {base}/{timeframe}/ohlcv_{timeframe}_{config_id}.parquet|.csv)
- --session: 'all' | 'regular'(기본 all) — 정규장만 남길지 여부
- --exchange-tz: 거래소 타임존(기본 America/New_York)
- --open/--close: 정규장 시작/종료 시각 문자열(기본 09:30 / 16:00)
- --base: 입력 기본 폴더 루트(기본 data/ohlcv)

> 사용되는 함수의 출력값 목록과 그 의미
- *_clean.parquet 또는 *_clean.csv: 정합화·정렬·필터링된 결과 파일
- 콘솔 로그: 입력/출력 경로, 행/열 수 변화, 제거/필터 건수 요약
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

def infer_input_path(cfg: dict, base_dir: str) -> Path:
    tf = cfg["timeframe"]
    cid = cfg["config_id"]
    cand = Path(base_dir) / tf / f"ohlcv_{tf}_{cid}.parquet"
    if cand.exists():
        return cand
    cand = cand.with_suffix(".csv")
    if cand.exists():
        return cand
    raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {cand}")

def load_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=[0])
    if df.empty:
        raise RuntimeError(f"데이터가 비어 있습니다: {path}")
    return df

def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("DatetimeIndex가 아닙니다. CSV 로드 시 parse_dates 설정을 확인하세요.")
    if idx.tz is None:
        df.index = idx.tz_localize("UTC")
    else:
        df.index = idx.tz_convert("UTC")
    return df

def _filter_regular_session(df: pd.DataFrame, exchange_tz: str, open_str: str, close_str: str) -> pd.DataFrame:
    # UTC→거래소 타임존으로 변환하여 시각 필터 후 다시 UTC로 복귀
    local = df.copy()
    local.index = local.index.tz_convert(exchange_tz)
    t = local.index.time
    open_h, open_m = map(int, open_str.split(":"))
    close_h, close_m = map(int, close_str.split(":"))
    mask = (
        (t >= pd.Timestamp(hour=open_h, minute=open_m).time()) &
        (t <= pd.Timestamp(hour=close_h, minute=close_m).time())
    )
    local = local[mask]
    local.index = local.index.tz_convert("UTC")
    return local

def clean_df(
    df: pd.DataFrame,
    session: str,
    exchange_tz: str,
    open_str: str,
    close_str: str,
    timeframe: str,
) -> tuple[pd.DataFrame, dict]:
    before_rows = len(df)
    df = _to_utc_index(df)
    # 중복 제거 + 정렬 + 빈행 제거
    dup_cnt = int(df.index.duplicated(keep="first").sum())
    if dup_cnt:
        df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    na_rows_before = before_rows
    df = df.dropna(how="all")
    na_removed = na_rows_before - len(df) - dup_cnt

    filtered = 0
    if session == "regular" and timeframe in {"1H", "15m", "5m"}:
        r_before = len(df)
        df = _filter_regular_session(df, exchange_tz, open_str, close_str)
        filtered = r_before - len(df)

    stats = {
        "duplicated_removed": dup_cnt,
        "all_na_rows_removed": max(0, na_removed),
        "session_filtered_rows": filtered,
        "rows_before": before_rows,
        "rows_after": len(df),
        "columns": df.shape[1],
        "start": str(df.index.min()) if len(df) else None,
        "end": str(df.index.max()) if len(df) else None,
    }
    return df, stats

def save_df(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path.with_suffix(".parquet"))
        return out_path.with_suffix(".parquet")
    except Exception:
        df.to_csv(out_path.with_suffix(".csv"), date_format="%Y-%m-%d %H:%M:%S%z")
        return out_path.with_suffix(".csv")

def main():
    p = argparse.ArgumentParser(description="OHLCV 클린·정렬(UTC/중복제거/정렬/(옵션)정규장)")
    p.add_argument("--config", default="data/config_snapshots/config.json")
    p.add_argument("--in", dest="in_path", default=None, help="입력 파일 경로(미지정 시 config로 추론)")
    p.add_argument("--base", default="data/ohlcv", help="입력 기본 폴더 루트")
    p.add_argument("--session", choices=["all", "regular"], default="all")
    p.add_argument("--exchange-tz", default="America/New_York")
    p.add_argument("--open", dest="open_str", default="09:30")
    p.add_argument("--close", dest="close_str", default="16:00")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    timeframe = cfg["timeframe"]
    cid = cfg["config_id"]

    if args.in_path:
        in_path = Path(args.in_path)
    else:
        in_path = infer_input_path(cfg, args.base)

    if not in_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {in_path}")

    print(f"[INFO] input={in_path} timeframe={timeframe} config_id={cid} session={args.session}")

    df = load_df(in_path)
    df_clean, stats = clean_df(
        df,
        session=args.session,
        exchange_tz=args.exchange_tz,
        open_str=args.open_str,
        close_str=args.close_str,
        timeframe=timeframe,
    )

    out_base = in_path.with_name(in_path.stem + "_clean")
    out_path = save_df(df_clean, out_base)

    print(f"[DONE] saved={out_path} rows_before={stats['rows_before']:,} rows_after={stats['rows_after']:,} "
          f"dup_removed={stats['duplicated_removed']:,} na_removed={stats['all_na_rows_removed']:,} "
          f"session_filtered={stats['session_filtered_rows']:,} "
          f"from={stats['start']} to={stats['end']}")

if __name__ == "__main__":
    main()

"""
UTC 정규화·정렬·정합성 검증·완전 조정(Adj OHLC)·스냅샷까지 수행하여 재현 가능한 백테스트 입력 데이터를 준비합니다.

이 파일의 목적:
- 시계열 시세 데이터의 타임존/정렬/중복/품질을 일관되게 정리하고, 필요 시 AdjClose 기반으로 O/H/L/C를 완전 조정합니다.
- 룩어헤드 금지 체결 정책을 위해 open 컬럼 존재를 보장하고, Parquet(+SHA-256, 버전) 스냅샷으로 재현성을 확보합니다.

사용되는 변수와 함수 목록:
- 변수
  - 없음
- 함수
  - _to_utc(ts: pandas.Series)
    - 역할: 입력 시각 열을 UTC 타임존으로 정규화
    - 입력값: ts: pandas.Series - 시각 데이터(문자열/타임스탬프 혼합 허용)
    - 반환값: pandas.DatetimeIndex - UTC 기준 타임스탬프
  - _dedup_sort_set_index(df: pandas.DataFrame, timestamp_col: str)
    - 역할: timestamp 정렬·중복 제거 후 인덱스 설정
    - 입력값: df: pandas.DataFrame, timestamp_col: str - 시각 컬럼명
    - 반환값: pandas.DataFrame - UTC 인덱스(timestamp)로 정렬/중복 제거된 프레임
  - _apply_calendar(df: pandas.DataFrame, drop_weekends: bool, holidays: Optional[Sequence])
    - 역할: 주말/휴일 제거로 거래 달력 정합성 적용
    - 입력값: drop_weekends: bool=False, holidays: Optional[Sequence]=None
    - 반환값: pandas.DataFrame - 캘린더가 적용된 프레임
  - _quality_gate(df: pandas.DataFrame)
    - 역할: OHLC 결측/음수/비정상 바 제거( low ≤ min(open,close,high) ≤ high )
    - 입력값: df: pandas.DataFrame
    - 반환값: pandas.DataFrame - 품질 게이트를 통과한 프레임
  - _full_adjust_ohlc(df: pandas.DataFrame)
    - 역할: AdjClose/Close 비율로 O/H/L/C 완전 조정 컬럼(open/high/low/close_adj) 생성
    - 입력값: df: pandas.DataFrame - close, adj_close 보유 시 동작
    - 반환값: pandas.DataFrame - 조정 컬럼이 추가된 프레임(조건부)
  - prepare_market_data(df: pandas.DataFrame, *, timestamp_col: str="timestamp", drop_weekends: bool=False, holidays: Optional[Sequence]=None, compute_adjusted: bool=True, cost_meta: Optional[CostMeta]=None)
    - 역할: 데이터 준비 파이프라인(정규화→캘린더→품질→조정) 실행 및 리포트/메타 반환
    - 입력값: 위 서명 참조
    - 반환값: (pandas.DataFrame, CostMeta, dict) - 정제 데이터, 비용 메타, 요약 리포트
  - snapshot_parquet(df: pandas.DataFrame, out_dir: pathlib.Path, *, name: str="snapshot", version: Optional[str]=None, engine: str="pyarrow")
    - 역할: 데이터 스냅샷 Parquet 저장 및 SHA-256 해시/버전 태깅
    - 입력값: df, out_dir, name, version, engine
    - 반환값: dict - {"path", "sha256", "version"}

파일의 흐름(→ / ->):
- 입력 정리(UTC 정규화·정렬·중복 제거) -> 거래 달력 적용(주말/휴일 제거) -> 품질 게이트(결측/음수/바 규칙) -> 완전 조정 OHLC(선택)
  -> open 컬럼 확인 -> (정제 데이터, 비용 메타, 리포트) 반환 -> Parquet 스냅샷 저장(SHA-256, 버전)
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class CostMeta:
    commission: float = 0.0
    slippage: float = 0.0
    lot_step: float = 1.0


def _to_utc(ts: pd.Series) -> pd.DatetimeIndex:
    return pd.to_datetime(ts, utc=True, errors="coerce")


def _dedup_sort_set_index(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    d = df.dropna(subset=[timestamp_col]).copy()
    d[timestamp_col] = _to_utc(d[timestamp_col])
    d = d.sort_values(timestamp_col).drop_duplicates(subset=[timestamp_col], keep="last")
    d = d.set_index(timestamp_col)
    d.index.name = "timestamp"
    return d


def _apply_calendar(
    df: pd.DataFrame, drop_weekends: bool, holidays: Optional[Sequence[object]]
) -> pd.DataFrame:
    d = df
    if drop_weekends:
        d = d[d.index.dayofweek < 5]
    if holidays:
        h = pd.DatetimeIndex(pd.to_datetime(holidays, utc=True)).normalize()
        d = d[~d.index.normalize().isin(h)]
    return d


def _quality_gate(df: pd.DataFrame) -> pd.DataFrame:
    req = ["open", "high", "low", "close"]
    d = df.dropna(subset=req)
    for c in req:
        d = d[d[c] > 0]
    mins = d[["open", "high", "close"]].min(axis=1)
    maxs = d[["open", "high", "close"]].max(axis=1)
    return d[(d["low"] <= mins) & (maxs <= d["high"])]


def _full_adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if "adj_close" not in df.columns:
        return df
    d = df.dropna(subset=["close", "adj_close"]).copy()
    a = d["adj_close"] / d["close"]
    d["open_adj"] = d["open"] * a
    d["high_adj"] = d["high"] * a
    d["low_adj"] = d["low"] * a
    d["close_adj"] = d["close"] * a
    return d


def prepare_market_data(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    drop_weekends: bool = False,
    holidays: Optional[Sequence[object]] = None,
    compute_adjusted: bool = True,
    cost_meta: Optional[CostMeta] = None,
) -> Tuple[pd.DataFrame, CostMeta, Dict[str, object]]:
    n_in = len(df)
    d = _dedup_sort_set_index(df, timestamp_col)
    d = _apply_calendar(d, drop_weekends=drop_weekends, holidays=holidays)
    d = _quality_gate(d)
    if compute_adjusted:
        d = _full_adjust_ohlc(d)
    if "open" not in d.columns:
        raise ValueError("open column required")
    cost = cost_meta or CostMeta()
    rep: Dict[str, object] = {
        "rows_in": n_in,
        "rows_out": len(d),
        "first_ts": d.index.min() if len(d) else None,
        "last_ts": d.index.max() if len(d) else None,
        "tz": "UTC",
        "has_adjusted": all(c in d.columns for c in ("open_adj", "high_adj", "low_adj", "close_adj")),
    }
    return d, cost, rep


def snapshot_parquet(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    name: str = "snapshot",
    version: Optional[str] = None,
    engine: str = "pyarrow",
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = df.reset_index().to_csv(index=False).encode("utf-8")
    digest = sha256(payload).hexdigest()
    ver = version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    fname = f"{name}_{digest[:8]}_v{ver}.parquet"
    path = out_dir / fname
    df.to_parquet(path, engine=engine)
    return {"path": str(path), "sha256": digest, "version": ver}

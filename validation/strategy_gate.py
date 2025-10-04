# validation/strategy_gate.py
"""
Strategy Rules Gate

의도:
- 전략 규약(타이밍/입력)을 문서대로 따랐는지 검증한다.
  * 신호: Close(t-1) 결정 → Open(t) 집행(룩어헤드 금지)
  * 스탑: Close(t) 판정 → Open(t+1) 집행
  * 충돌 시 Stop > Signal
  * SMA 입력은 close_adj(우선, 부재 시 close)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd


class GateResult(TypedDict):
    passed: bool
    errors: List[str]
    warnings: List[str]
    evidence: Dict[str, Any]


@dataclass
class Artifacts:
    """검증에 필요한 산출물 핸들."""
    snapshot_parquet_path: str
    run_meta: Dict[str, Any]
    trades: pd.DataFrame  # columns: ts, reason, ...


# --------- 유틸 ---------

def _to_utc_ts(x: Any) -> pd.Timestamp:
    """임의의 시간 값을 UTC Timestamp로 표준화."""
    if isinstance(x, pd.Timestamp):
        return x.tz_convert("UTC") if x.tz is not None else x.tz_localize("UTC")
    if isinstance(x, (str, bytes)):
        return pd.to_datetime(x, utc=True)
    if isinstance(x, datetime):
        return pd.Timestamp(x, tz="UTC") if x.tzinfo is None else pd.Timestamp(x).tz_convert("UTC")
    return pd.to_datetime(x, utc=True)


def _pick_col(df: pd.DataFrame, base: str) -> str:
    """*_adj 우선 컬럼 선택."""
    c_adj = f"{base}_adj"
    return c_adj if c_adj in df.columns else base


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """인덱스를 DatetimeIndex[UTC]로 표준화."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("snapshot index must be DatetimeIndex")
    return df.tz_localize("UTC") if df.index.tz is None else df.tz_convert("UTC")


# --------- 핵심 검증 ---------

def run(artifacts: Artifacts) -> GateResult:
    errors: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {}

    # 0) 입력 방어
    if artifacts.trades is None or artifacts.trades.empty:
        return GateResult(passed=False, errors=["[trades:missing] trades가 비어 있습니다."], warnings=[], evidence={})
    if not isinstance(artifacts.run_meta, dict):
        return GateResult(passed=False, errors=["[run_meta:missing] run_meta가 필요합니다."], warnings=[], evidence={})

    # 1) 스냅샷 로드 & 기준 컬럼
    try:
        df = pd.read_parquet(artifacts.snapshot_parquet_path)
    except Exception as e:  # pandas/엔진별 예외 다양 → 단일 메시지로 즉시 실패
        return GateResult(passed=False, errors=[f"[parquet:read] 스냅샷 로드 실패: {e}"], warnings=[], evidence={})
    df = _ensure_utc_index(df)

    close_col = _pick_col(df, "close")
    low_col = _pick_col(df, "low")
    open_col = _pick_col(df, "open")  # 증적용

    # 2) 파라미터 추출 (SMA 10/50 고정)
    params = artifacts.run_meta.get("params", {})
    N = int(params.get("N", 20))
    epsilon = float(params.get("epsilon", 0.0))
    sma_short, sma_long = 10, 50

    # 3) SMA/Donchian 재계산(룩어헤드 금지 시프트 포함)
    close_s = pd.to_numeric(df[close_col], errors="coerce")
    low_s = pd.to_numeric(df[low_col], errors="coerce")

    sma_s = close_s.rolling(sma_short, min_periods=sma_short).mean()
    sma_l = close_s.rolling(sma_long, min_periods=sma_long).mean()
    decided = (sma_s - sma_l) > epsilon           # Close(t)에서의 결정값
    signal_next = decided.astype("boolean").shift(1).fillna(False)  # 다음날 Open(t+1) 집행 플래그

    prev_low_tminus1 = low_s.shift(1).rolling(N, min_periods=N).min()
    stop_hit_t = (low_s <= prev_low_tminus1).fillna(False)  # Close(t) 판정

    # 4) SMA 입력 정의 확인(close_adj 우선)
    price_used = artifacts.run_meta.get("price_columns_used", {})
    used_close = price_used.get("close")
    if "close_adj" in df.columns and used_close != "close_adj":
        errors.append(f"[sma:input] close_adj가 존재하지만 run_meta.price_columns_used.close={used_close!r}")

    # 5) trades 정규화(UTC, 안정 정렬)
    trades = artifacts.trades.copy()
    if "ts" not in trades.columns or "reason" not in trades.columns:
        return GateResult(passed=False, errors=["[trades:schema] 'ts'와 'reason' 컬럼이 필요합니다."], warnings=[], evidence={})
    trades["ts"] = trades["ts"].map(_to_utc_ts)
    trades.sort_values(["ts"], kind="stable", inplace=True)
    trades.reset_index(drop=True, inplace=True)

    # 6) 신호 검증: 모든 signal 체결은 signal_next(True)인 날이어야 함
    signal_trades = trades[trades["reason"] == "signal"]
    bad_signals: List[str] = []
    early_window_signals: List[str] = []
    for _, tr in signal_trades.iterrows():
        ts: pd.Timestamp = tr["ts"]
        if ts not in df.index:
            bad_signals.append(f"{ts.isoformat()} — 스냅샷 인덱스에 없음")
            continue
        if not bool(signal_next.loc[ts]):
            bad_signals.append(f"{ts.isoformat()} — signal_next=False/NA")
        # 초기 윈도: 결정 시계열의 shift(1)가 NaN이면 의사결정 금지 구간
        if pd.isna(decided.shift(1).loc[ts]):
            early_window_signals.append(f"{ts.isoformat()} — 초기 윈도 구간 체결 발생")

    if bad_signals:
        errors.append("[signal:timing] Close(t-1) 결정 → Open(t) 집행 위반:\n  - " + "\n  - ".join(bad_signals))
    if early_window_signals:
        errors.append("[signal:window] SMA 윈도 경계 구간에서 신호 체결 발생:\n  - " + "\n  - ".join(early_window_signals))

    # 7) 스탑 검증: 모든 stop 체결은 전일(t-1) stop_hit(t-1)=True였어야 함
    stop_trades = trades[trades["reason"] == "stop"]
    bad_stops: List[str] = []
    for _, tr in stop_trades.iterrows():
        ts: pd.Timestamp = tr["ts"]
        if ts not in df.index:
            bad_stops.append(f"{ts.isoformat()} — 스냅샷 인덱스에 없음")
            continue
        pos = df.index.get_loc(ts)
        if isinstance(pos, slice):
            pos = pos.start
        if pos == 0:
            bad_stops.append(f"{ts.isoformat()} — 시계열 시작일에는 stop 체결 불가")
            continue
        ts_prev = df.index[pos - 1]
        if not bool(stop_hit_t.loc[ts_prev]):
            bad_stops.append(f"{ts.isoformat()} — 전일({ts_prev.isoformat()}) stop_hit=False")

    if bad_stops:
        errors.append("[stop:timing] Close(t) 판정 → Open(t+1) 집행 위반:\n  - " + "\n  - ".join(bad_stops))

    # 8) Stop > Signal: 동일 시각 체결 순서(정렬은 stable)
    order_violations: List[str] = []
    for ts, g in trades.groupby("ts"):
        reasons = list(g["reason"])
        if "stop" in reasons and "signal" in reasons:
            if reasons[0] != "stop":
                order_violations.append(f"{ts.isoformat()} — 체결 순서: {reasons}")
    if order_violations:
        errors.append("[priority] Stop > Signal 위반(동일 시각):\n  - " + "\n  - ".join(order_violations))

    evidence.update(
        {
            "close_col_used": close_col,
            "low_col_used": low_col,
            "open_col_used": open_col,
            "params": {"N": N, "epsilon": epsilon, "sma_short": sma_short, "sma_long": sma_long},
            "n_signal_trades": int(len(signal_trades)),
            "n_stop_trades": int(len(stop_trades)),
        }
    )

    return GateResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
    )

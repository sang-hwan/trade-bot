# validation/strategy_gate.py
"""
Strategy Rules Gate

의도:
- 전략 규약(타이밍/입력)을 문서대로 따랐는지 검증한다.
  * 신호: Close(t-1) 결정 → Open(t) 집행(룩어헤드 금지)
  * 스탑: Close(t) 판정 → Open(t+1) 집행
  * 충돌 시 Stop > Signal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict

import pandas as pd

from strategy.signals import sma_cross_long_only
from strategy.stops import donchian_stop_long


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
    trades: pd.DataFrame


def run(artifacts: Artifacts) -> GateResult:
    """전략 규칙(타이밍, 우선순위) 검증의 단일 진입점."""
    errors: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {}

    # 입력 데이터 유효성 검사
    if artifacts.trades is None or artifacts.trades.empty:
        # 거래가 없는 것은 오류가 아니므로 통과 처리
        return GateResult(passed=True, errors=[], warnings=["[trades:empty] trades가 비어 있어 검증을 건너뜁니다."], evidence={})
    if not isinstance(artifacts.run_meta, dict):
        return GateResult(passed=False, errors=["[run_meta:missing] run_meta가 필요합니다."], warnings=[], evidence={})

    # 스냅샷 데이터 로드
    try:
        df = pd.read_parquet(artifacts.snapshot_parquet_path)
    except Exception as e:
        errors.append(f"[parquet:read] 스냅샷 로드 실패: {e}")
        return GateResult(passed=False, errors=errors, warnings=warnings, evidence=evidence)

    # 파라미터 추출
    params = artifacts.run_meta.get("params", {})
    N = int(params.get("N", 20))
    epsilon = float(params.get("epsilon", 0.0))
    sma_short = int(params.get("sma_short", 10))
    sma_long = int(params.get("sma_long", 50))
    
    # 공식 strategy 모듈을 사용하여 기대 신호/스탑 재계산
    expected_signals = sma_cross_long_only(df, short=sma_short, long=sma_long, epsilon=epsilon)
    expected_stops = donchian_stop_long(df, N=N)

    # 거래 내역(trades) 정규화
    trades = artifacts.trades.copy()
    trades["ts"] = pd.to_datetime(trades["ts"], utc=True)
    trades.sort_values(["ts", "reason"], ascending=[True, True], kind="stable", inplace=True) # stop이 signal보다 먼저 오도록 정렬

    # 신호(Signal) 타이밍 검증
    signal_trades = trades[trades["reason"] == "signal"]
    for _, trade in signal_trades.iterrows():
        ts = trade["ts"]
        if ts not in expected_signals.index or not expected_signals[ts]:
            errors.append(f"[signal:timing] {ts}: 예기치 않은 'signal' 거래가 발생했습니다 (기대치: False).")

    # 스탑(Stop) 타이밍 검증
    stop_trades = trades[trades["reason"] == "stop"]
    for _, trade in stop_trades.iterrows():
        ts = trade["ts"]
        if ts not in df.index:
            errors.append(f"[stop:timing] {ts}: 거래 시간이 스냅샷에 존재하지 않습니다.")
            continue
        
        pos = df.index.get_loc(ts)
        if pos == 0:
            errors.append(f"[stop:timing] {ts}: 시계열 첫 날에는 스탑 거래가 발생할 수 없습니다.")
            continue
            
        ts_prev = df.index[pos - 1]
        if ts_prev not in expected_stops.index or not expected_stops.loc[ts_prev, "stop_hit"]:
            errors.append(f"[stop:timing] {ts}: 전일({ts_prev.date()})에 스탑 조건이 충족되지 않았습니다.")

    # Stop > Signal 우선순위 검증
    for ts, group in trades.groupby("ts"):
        reasons = group["reason"].tolist()
        if "stop" in reasons and "signal" in reasons:
            # 정렬 규칙에 의해 stop이 항상 먼저 와야 함
            if reasons.index("stop") > reasons.index("signal"):
                errors.append(f"[priority] {ts}: Stop > Signal 우선순위 위반. 체결 순서: {reasons}")

    evidence.update({
        "params_validated": {"N": N, "epsilon": epsilon, "sma_short": sma_short, "sma_long": sma_long},
        "n_signal_trades_validated": len(signal_trades),
        "n_stop_trades_validated": len(stop_trades),
    })

    return GateResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
    )

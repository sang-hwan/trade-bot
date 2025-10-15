# validation/simulation_gate.py
"""
Simulation Timeline & Accounting Gate

의도:
- 포트폴리오 엔진의 집행/라운딩/비용/회계 등식이 문서 규약대로 계산되는지 검증한다.
  * 체결가: Open_eff = open_adj(우선) × (1±slip)
  * 회계 등식: 매일 [현금 변화 + 포지션 평가손익 − 비용] = 자본 변화
  * 산출물 계약: trades, metrics, equity_curve 정합성
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict
import math

import numpy as np
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
    snapshot_meta: Dict[str, Any]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: Dict[str, Any]


# ---------- 유틸 ----------

def _is_multiple(value: float, step: float, tol: float = 1e-9) -> bool:
    """value가 step의 배수인지 확인한다."""
    if step <= 0:
        return True
    q = value / step
    return math.isclose(q, round(q), rel_tol=0.0, abs_tol=tol)


def _aeq(a: float, b: float, atol: float) -> bool:
    """두 부동소수점 값이 절대 허용 오차 내에서 같은지 확인한다."""
    return math.isclose(a, b, rel_tol=0.0, abs_tol=atol)


# ---------- 핵심 검증 ----------

def run(artifacts: Artifacts) -> GateResult:
    """시뮬레이션 타임라인·라운딩·회계 등식 검증의 단일 진입점."""
    errors: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {}

    # 입력 데이터 로드 및 파라미터 추출
    if artifacts.equity_curve is None or artifacts.equity_curve.empty:
        errors.append("[equity_curve:missing] equity_curve가 비어 있습니다.")
    if not isinstance(artifacts.run_meta, dict) or not isinstance(artifacts.metrics, dict):
        errors.append("[meta:missing] run_meta/metrics가 필요합니다.")
    if errors:
        return GateResult(passed=False, errors=errors, warnings=warnings, evidence=evidence)

    try:
        prices = pd.read_parquet(artifacts.snapshot_parquet_path)
    except Exception as e:
        errors.append(f"[parquet:read] 스냅샷 로드 실패: {e}")
        return GateResult(passed=False, errors=errors, warnings=warnings, evidence=evidence)

    # 단일 자산 백테스트이므로, snapshot_meta에서 심볼 이름을 가져옵니다.
    sym = artifacts.snapshot_meta.get("symbol")
    if not sym:
        errors.append("[meta:symbol] snapshot_meta에서 'symbol'을 찾을 수 없습니다.")
        return GateResult(passed=False, errors=errors, warnings=warnings, evidence={})
    symbols = [sym]
    params = artifacts.run_meta.get("params", {})
    price_step = float(params.get("price_step", 0.0))
    lot_step = float(params.get("lot_step", 1.0))
    commission_rate = float(params.get("commission_rate", 0.0))
    slip = float(params.get("slip", 0.0))
    initial_equity = float(artifacts.metrics.get("initial_equity", 0.0))
    price_cols_used = artifacts.run_meta.get("price_columns_used", {})
    open_col = price_cols_used.get("open", "open_adj")
    close_col = price_cols_used.get("close", "close_adj")
    # 거래 내역(trades) 검증
    trades = artifacts.trades.copy()
    if not trades.empty:
        trades["ts"] = pd.to_datetime(trades["ts"], utc=True)
        trades.sort_values(["ts"], kind="stable", inplace=True)

        for _, tr in trades.iterrows():
            ts, sym, side, qty = tr["ts"], tr["symbol"], tr["side"], tr["qty"]
            if ts not in prices.index:
                errors.append(f"[trades:ts] 거래 시간({ts})이 스냅샷에 존재하지 않습니다.")
                continue

            if not _is_multiple(qty, lot_step):
                errors.append(f"[qty:lot_step] {ts} {sym}: 수량({qty})이 lot_step({lot_step})의 배수가 아닙니다.")

            open_price = prices.loc[ts, open_col]
            signed_slip = (1.0 + slip) if side == "buy" else (1.0 - slip)
            expected_px_approx = open_price * signed_slip
            if not _aeq(tr["price"], expected_px_approx, atol=price_step + 1e-9):
                 warnings.append(f"[price:open_eff] {ts} {sym}: 체결가({tr['price']})가 기대치({expected_px_approx:.4f})와 차이가 있습니다.")

            expected_comm = abs(tr["price"] * qty) * commission_rate
            if not _aeq(tr["commission"], expected_comm, atol=1e-9):
                errors.append(f"[commission] {ts} {sym}: 수수료({tr['commission']})가 기대치({expected_comm})와 다릅니다.")

    # 회계 장부 재구성
    cash = initial_equity
    positions: Dict[str, float] = {s: 0.0 for s in symbols}
    avg_entries: Dict[str, float] = {s: 0.0 for s in symbols}
    reconstructed_equity_series = []
    
    trades_by_ts = trades.groupby("ts") if not trades.empty else {}

    for ts in prices.index:
        if ts in trades_by_ts.groups:
            for _, tr in trades_by_ts.get_group(ts).iterrows():
                sym, side, qty, price, comm = tr["symbol"], tr["side"], tr["qty"], tr["price"], tr["commission"]
                if side == "buy":
                    cash -= price * qty + comm
                    total_cost_prev = avg_entries[sym] * positions[sym]
                    positions[sym] += qty
                    avg_entries[sym] = (total_cost_prev + price * qty) / positions[sym]
                else: # sell
                    cash += price * qty - comm
                    positions[sym] -= qty
                    if positions[sym] == 0:
                        avg_entries[sym] = 0.0
        
        pos_value = sum(positions[s] * prices.loc[ts, close_col] for s in symbols if pd.notna(prices.loc[ts, close_col]))
        reconstructed_equity_series.append(cash + pos_value)

    # equity_curve 대조
    ec_to_check = artifacts.equity_curve.copy()
    ec_to_check.index = pd.to_datetime(ec_to_check.index, utc=True)
    reconstructed_df = pd.DataFrame({"equity_recon": reconstructed_equity_series}, index=prices.index)
    
    comparison = pd.merge(ec_to_check, reconstructed_df, left_index=True, right_index=True, how="left")
    mismatches = comparison[~np.isclose(comparison["equity"], comparison["equity_recon"], atol=1e-6)]

    for ts, row in mismatches.iterrows():
        errors.append(f"[equity_curve] {ts}: 재구성된 자산({row['equity_recon']:.4f})이 보고된 자산({row['equity']:.4f})과 다릅니다.")
    
    # 최종 지표(metrics) 검증
    reconstructed_final_equity = reconstructed_equity_series[-1] if reconstructed_equity_series else initial_equity
    final_equity_metric = artifacts.metrics.get("final_equity")
    if final_equity_metric is not None and not _aeq(reconstructed_final_equity, final_equity_metric, atol=1e-6):
        errors.append(f"[metrics:final_equity] 최종 자산({reconstructed_final_equity:.4f})이 보고된 지표({final_equity_metric:.4f})와 다릅니다.")

    # run_meta 검증
    if "params" not in artifacts.run_meta:
        errors.append("[run_meta:params] 'params' 키가 run_meta에 존재해야 합니다.")
    
    evidence.update({
        "reconstructed_final_equity": reconstructed_final_equity,
        "n_trades_validated": len(trades),
        "params_validated": params,
    })

    return GateResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
    )

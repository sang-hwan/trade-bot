# validation/simulation_gate.py
"""
Simulation Timeline & Accounting Gate

의도:
- 집행/라운딩/비용/회계 등식이 문서 규약대로 계산되는지 검증한다.
  * 체결가: Open_eff = open_adj(우선, 없으면 open) × (1±slip)
  * 라운딩: 가격=price_step 최근접(동률 내림), 수량=lot_step 하향 배수
  * 회계 등식: 매일 [현금 변화 + 포지션 평가손익 − 비용] = 자본 변화
  * 산출물 계약: trades.reason/side, metrics·equity_curve 정합성
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict
import math

import pandas as pd


class GateResult(TypedDict):
    passed: bool
    errors: List[str]
    warnings: List[str]
    evidence: Dict[str, Any]


@dataclass
class Artifacts:
    snapshot_parquet_path: str
    run_meta: Dict[str, Any]
    snapshot_meta: Dict[str, Any]
    trades: pd.DataFrame           # columns: ts, side, reason, qty, price, commission, slip, realized_pnl, ...
    equity_curve: pd.DataFrame     # columns: equity (index: ts)
    metrics: Dict[str, Any]


# ---------- 유틸 ----------

_ALLOWED_REASONS = {"signal", "stop", "rebalance"}
_ALLOWED_SIDES = {"buy", "sell"}


def _pick_col(df: pd.DataFrame, base: str) -> str:
    c_adj = f"{base}_adj"
    return c_adj if c_adj in df.columns else base


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("snapshot index must be DatetimeIndex")
    return df.tz_localize("UTC") if df.index.tz is None else df.tz_convert("UTC")


def _round_price_nearest_tie_down(x: float, step: float) -> float:
    if step <= 0:
        return x
    n = x / step
    down = math.floor(n)
    up = math.ceil(n)
    # tie → down
    if abs(n - down) < abs(up - n) or math.isclose(n - down, up - n, rel_tol=0.0, abs_tol=1e-15):
        return down * step
    return up * step


def _is_multiple(value: float, step: float, tol: float = 1e-12) -> bool:
    if step <= 0:
        return True
    q = value / step
    return math.isclose(q, round(q), rel_tol=0.0, abs_tol=tol)


def _aeq(a: float, b: float, atol: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=atol)


# ---------- 핵심 검증 ----------

def run(artifacts: Artifacts) -> GateResult:
    """시뮬레이션 타임라인·라운딩·회계 등식 검증의 단일 진입점."""
    errors: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {}

    # 0) 입력 방어
    if artifacts.trades is None or artifacts.trades.empty:
        return GateResult(passed=False, errors=["[trades:missing] trades가 비어 있습니다."], warnings=[], evidence={})
    if artifacts.equity_curve is None or artifacts.equity_curve.empty:
        return GateResult(passed=False, errors=["[equity_curve:missing] equity_curve가 비어 있습니다."], warnings=[], evidence={})
    if not isinstance(artifacts.run_meta, dict) or not isinstance(artifacts.metrics, dict):
        return GateResult(passed=False, errors=["[meta:missing] run_meta/metrics가 필요합니다."], warnings=[], evidence={})

    # 1) 스냅샷 로드 & 기준 컬럼
    try:
        df = pd.read_parquet(artifacts.snapshot_parquet_path)
    except Exception as e:  # 엔진별 예외 다양 → 즉시 실패
        return GateResult(passed=False, errors=[f"[parquet:read] 스냅샷 로드 실패: {e}"], warnings=[], evidence={})
    df = _ensure_utc_index(df)
    open_col = _pick_col(df, "open")
    close_col = _pick_col(df, "close")

    # 2) 파라미터
    params = artifacts.run_meta.get("params", {})
    price_step = float(params.get("price_step", 0.0))
    lot_step = float(params.get("lot_step", 1.0))
    commission_rate = float(params.get("commission_rate", 0.0))
    slip = float(params.get("slip", 0.0))
    initial_equity = float(artifacts.metrics.get("initial_equity", artifacts.run_meta.get("initial_equity", 0.0)))

    # 3) 산출물 계약(스키마/값 범위)
    bad_reason = set(artifacts.trades["reason"].dropna().unique()) - _ALLOWED_REASONS
    if bad_reason:
        errors.append(f"[trades:reason] 허용되지 않은 reason 값: {sorted(bad_reason)}")
    if "side" in artifacts.trades:
        bad_side = set(artifacts.trades["side"].dropna().unique()) - _ALLOWED_SIDES
        if bad_side:
            errors.append(f"[trades:side] 허용되지 않은 side 값: {sorted(bad_side)}")
    if "equity" not in artifacts.equity_curve.columns:
        errors.append("[equity_curve:schema] 'equity' 컬럼 필요")

    # 4) trades 정규화(UTC, 안정 정렬)
    trades = artifacts.trades.copy()
    if "ts" not in trades.columns:
        return GateResult(passed=False, errors=["[trades:schema] 'ts' 컬럼이 필요합니다."], warnings=[], evidence={})
    trades["ts"] = pd.to_datetime(trades["ts"], utc=True)
    trades.sort_values(["ts"], kind="stable", inplace=True)
    trades.reset_index(drop=True, inplace=True)

    # 5) 가격·수량 라운딩 및 체결가 정의 검증
    price_mismatch: List[str] = []
    qty_mismatch: List[str] = []
    comm_mismatch: List[str] = []

    for _, tr in trades.iterrows():
        ts = tr["ts"]
        if ts not in df.index:
            errors.append(f"[trades:ts] 스냅샷 인덱스에 없는 ts: {ts.isoformat()}")
            continue

        # 기대 체결가: Open_eff = open*(1±slip) → price_step 라운딩
        o = float(pd.to_numeric(df.loc[ts, open_col], errors="coerce"))
        if not math.isfinite(o):
            errors.append(f"[open:NaN] {ts.isoformat()} — open 값 비정상")
            continue

        side = str(tr.get("side", "")).lower()
        if side not in _ALLOWED_SIDES:
            errors.append(f"[trades:side] {ts.isoformat()} — side={tr.get('side')!r}")
            continue

        signed = (1.0 + slip) if side == "buy" else (1.0 - slip)
        expected_px = _round_price_nearest_tie_down(o * signed, price_step)
        got_px = float(tr.get("price", float("nan")))
        if not _aeq(expected_px, got_px, atol=max(1e-9, price_step * 1e-9)):
            price_mismatch.append(f"{ts.isoformat()} — expected={expected_px} got={got_px}")

        # 수량 라운딩
        qty = float(tr.get("qty", float("nan")))
        if not math.isfinite(qty) or qty <= 0:
            qty_mismatch.append(f"{ts.isoformat()} — qty 비정상: {qty}")
        elif not _is_multiple(qty, lot_step, tol=1e-12):
            qty_mismatch.append(f"{ts.isoformat()} — qty {qty} not multiple of lot_step {lot_step}")

        # 수수료 계산 일치
        expected_comm = abs(got_px * qty) * commission_rate
        got_comm = float(tr.get("commission", 0.0))
        if not _aeq(expected_comm, got_comm, atol=max(1e-9, expected_comm * 1e-9)):
            comm_mismatch.append(f"{ts.isoformat()} — commission expected={expected_comm} got={got_comm}")

    if price_mismatch:
        errors.append("[price:open_eff] 체결가(Open_eff) 불일치:\n  - " + "\n  - ".join(price_mismatch[:20]))
    if qty_mismatch:
        errors.append("[qty:lot_step] 수량 라운딩 위반:\n  - " + "\n  - ".join(qty_mismatch[:20]))
    if comm_mismatch:
        errors.append("[commission] 수수료 계산 불일치:\n  - " + "\n  - ".join(comm_mismatch[:20]))

    # 6) 일/일 회계 등식 + equity_curve 정합성
    eq = []
    cash = float(initial_equity)
    pos = 0.0
    trades_by_ts = trades.groupby("ts", sort=True)

    ec = artifacts.equity_curve.copy()
    ec.index = pd.to_datetime(ec.index, utc=True)

    for ts in df.index:  # 스냅샷 타임라인 기준
        if ts in trades_by_ts.indices:
            g = trades.loc[trades_by_ts.indices[ts]]
            for _, tr in g.iterrows():
                px = float(tr["price"])
                q = float(tr["qty"])
                comm = float(tr.get("commission", 0.0))
                if str(tr["side"]).lower() == "buy":
                    cash -= px * q + comm
                    pos += q
                else:  # sell
                    cash += px * q - comm
                    pos -= q
        # Close 평가
        c = float(pd.to_numeric(df.loc[ts, close_col], errors="coerce"))
        if not math.isfinite(c):
            errors.append(f"[close:NaN] {ts.isoformat()} — close 값 비정상")
            continue
        eq_t = cash + pos * c
        eq.append((ts, eq_t))

        if ts in ec.index:
            got_eq = float(ec.loc[ts, "equity"])
            if not _aeq(eq_t, got_eq, atol=max(1e-6, got_eq * 1e-9)):
                errors.append(f"[equity_curve] {ts.isoformat()} — expected={eq_t} got={got_eq}")

    if not eq:
        errors.append("[equity:empty] 재구성된 equity 시계열이 비어 있음")
    else:
        last_ts, last_eq = eq[-1]
        final_equity = float(artifacts.metrics.get("final_equity", float("nan")))
        if math.isfinite(final_equity) and not _aeq(last_eq, final_equity, atol=max(1e-6, final_equity * 1e-9)):
            errors.append(f"[metrics:final_equity] expected={last_eq} got={final_equity}")

        total_return = float(artifacts.metrics.get("total_return", float("nan")))
        if initial_equity > 0:
            expected_ret = last_eq / initial_equity - 1.0
            if math.isfinite(total_return) and not _aeq(expected_ret, total_return, atol=1e-12):
                errors.append(f"[metrics:total_return] expected={expected_ret} got={total_return}")

        eq_series = pd.Series([v for _, v in eq], index=[t for t, _ in eq])
        peak = eq_series.cummax()
        dd = eq_series / peak - 1.0
        mdd_calc = float(dd.min()) if not dd.empty else 0.0
        mdd_report = float(artifacts.metrics.get("mdd", float("nan")))
        if math.isfinite(mdd_report) and not _aeq(mdd_calc, mdd_report, atol=1e-12):
            errors.append(f"[metrics:mdd] expected={mdd_calc} got={mdd_report}")

    evidence.update(
        {
            "open_col_used": open_col,
            "close_col_used": close_col,
            "params": {
                "price_step": price_step,
                "lot_step": lot_step,
                "commission_rate": commission_rate,
                "slip": slip,
                "initial_equity": initial_equity,
            },
            "n_trades": int(len(trades)),
        }
    )

    return GateResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
    )

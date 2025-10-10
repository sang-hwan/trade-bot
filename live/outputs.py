# live/outputs.py
"""
목적: 실매매 결과를 시뮬레이션과 동일 스키마로 정리/기록.
산출물: trades, equity_curve, metrics, run_meta (+선택: 주문/체결 로그)

시간 규약: 모든 타임스탬프는 UTC ISO-8601("...Z").
파일 출력: out_dir 제공 시 CSV/JSON/JSONL로 기록(표준 라이브러리 사용).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional
from datetime import datetime, timezone
import csv
import json
import os
import math

# 시뮬레이션 모듈 재사용(가능 시), 불가 시 로컬 계산으로 대체.
try:
    from simulation.outputs import (  # type: ignore
        make_trades_df as _sim_make_trades_df,
        make_equity_curve as _sim_make_equity_curve_df,
        compute_metrics as _sim_compute_metrics,
    )
except ImportError:
    _sim_make_trades_df = None
    _sim_make_equity_curve_df = None
    _sim_compute_metrics = None


# ---------- 유틸 ----------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_float(x) -> float:
    return float(x)


def _ensure_ts(x: str) -> str:
    return str(x).strip()


def _sorted_by_ts(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: r.get("ts") or r.get("ts_utc") or "")


# ---------- 정규화 빌더 ----------

def make_trades(trades: Iterable[Mapping[str, object]]) -> list[dict]:
    """
    trades 정규화:
    - 필수 키: ts, symbol, side, qty, price, commission, reason
    - 정렬: ts 오름차순
    """
    if _sim_make_trades_df is not None:
        df = _sim_make_trades_df(trades)  # pandas.DataFrame(ts 오름차순)
        out = df.to_dict(orient="records")  # type: ignore[attr-defined]
        norm: list[dict] = []
        for d in out:
            d = dict(d)
            d["ts"] = _ensure_ts(d.get("ts") or d.get("ts_utc", ""))
            if "symbol" in d:
                d["symbol"] = str(d["symbol"])
            if "side" in d:
                d["side"] = str(d["side"]).lower()
            if "qty" in d:
                d["qty"] = _as_float(d["qty"])
            if "price" in d:
                d["price"] = _as_float(d["price"])
            d["commission"] = _as_float(d.get("commission", 0.0))
            if "reason" in d:
                d["reason"] = str(d["reason"]).lower()
            norm.append(d)
        return _sorted_by_ts(norm)

    # 로컬 fallback
    out: list[dict] = []
    for r in trades:
        d = dict(r)
        d["ts"] = _ensure_ts(d.get("ts") or d.get("ts_utc", ""))
        d["symbol"] = str(d["symbol"])
        d["side"] = str(d["side"]).lower()
        d["qty"] = _as_float(d["qty"])
        d["price"] = _as_float(d["price"])
        d["commission"] = _as_float(d.get("commission", 0.0))
        d["reason"] = str(d.get("reason", "rebalance")).lower()
        out.append(d)
    return _sorted_by_ts(out)


def make_equity_curve(points: Iterable[Mapping[str, object]]) -> list[dict]:
    """
    equity_curve 정규화:
    - 원소: {"ts": "...Z", "equity": float}
    - 정렬: ts 오름차순
    """
    if _sim_make_equity_curve_df is not None:
        ec = _sim_make_equity_curve_df(points)  # DataFrame(index=ts, col 'equity')
        if getattr(ec, "empty", True):
            return []
        ec_reset = ec.reset_index()
        return _sorted_by_ts([
            {"ts": _ensure_ts(str(ts)), "equity": _as_float(eq)}
            for ts, eq in zip(ec_reset["ts"], ec_reset["equity"])  # type: ignore[index]
        ])

    # 로컬 fallback
    out: list[dict] = []
    for p in points:
        ts = _ensure_ts(p.get("ts", ""))  # type: ignore[arg-type]
        eq = _as_float(p["equity"])        # type: ignore[index]
        out.append({"ts": ts, "equity": eq})
    return _sorted_by_ts(out)


def compute_metrics(equity_curve: list[Mapping[str, object]], trades: list[Mapping[str, object]]) -> dict[str, object]:
    """기본 지표 계산: 시뮬레이션 모듈 우선, 불가 시 로컬 계산."""
    # 시뮬레이션 경로(우선)
    if _sim_compute_metrics is not None and _sim_make_equity_curve_df is not None and _sim_make_trades_df is not None:
        ec_df = _sim_make_equity_curve_df(equity_curve)  # type: ignore[misc]

        # 빈 equity_curve → 안전 반환
        if getattr(ec_df, "empty", True):
            return {
                "initial_equity": 0.0,
                "final_equity": 0.0,
                "total_return": 0.0,
                "mdd": 0.0,
                "mdd_start_ts": None,
                "mdd_end_ts": None,
                "trade_count": len(trades),
                "win_rate": None,
                "payoff": None,
                "profit_factor": None,
            }

        initial_equity = float(ec_df["equity"].iloc[0])  # type: ignore[index]
        final_equity = float(ec_df["equity"].iloc[-1])   # type: ignore[index]

        # 초기자본 0/음수/NaN → 0으로 안전 처리(0으로 나누기 방지)
        if initial_equity <= 0.0 or math.isnan(initial_equity):
            return {
                "initial_equity": initial_equity,
                "final_equity": final_equity,
                "total_return": 0.0,
                "mdd": 0.0,
                "mdd_start_ts": None,
                "mdd_end_ts": None,
                "trade_count": len(trades),
                "win_rate": None,
                "payoff": None,
                "profit_factor": None,
            }

        trades_df = _sim_make_trades_df(trades)  # type: ignore[misc]
        sim_m = _sim_compute_metrics(ec_df, initial_equity=initial_equity, trades_df=trades_df)  # type: ignore[misc]
        return {
            "initial_equity": initial_equity,
            "final_equity": final_equity,
            "total_return": float(sim_m.get("total_return", 0.0)),   # type: ignore[call-arg]
            "mdd": float(sim_m.get("mdd", 0.0)),                     # type: ignore[call-arg]
            "mdd_start_ts": None,
            "mdd_end_ts": None,
            "trade_count": int(sim_m.get("trades", 0)),             # type: ignore[call-arg]
            "win_rate": sim_m.get("win_rate"),                      # type: ignore[call-arg]
            "payoff": sim_m.get("payoff"),                          # type: ignore[call-arg]
            "profit_factor": sim_m.get("profit_factor"),            # type: ignore[call-arg]
        }

    # 로컬 fallback
    if not equity_curve:
        return {
            "initial_equity": 0.0,
            "final_equity": 0.0,
            "total_return": 0.0,
            "mdd": 0.0,
            "mdd_start_ts": None,
            "mdd_end_ts": None,
            "trade_count": len(trades),
            "win_rate": None,
            "payoff": None,
            "profit_factor": None,
        }

    ecs = make_equity_curve(equity_curve)
    eq_vals = [float(r["equity"]) for r in ecs]
    ts_vals = [str(r["ts"]) for r in ecs]
    initial, final = eq_vals[0], eq_vals[-1]
    total_ret = (final / initial - 1.0) if initial > 0 else 0.0

    peak = eq_vals[0]
    mdd, mdd_start, mdd_end, peak_idx = 0.0, ts_vals[0], ts_vals[0], 0
    for i, v in enumerate(eq_vals):
        if v > peak:
            peak, peak_idx = v, i
        dd = (v / peak - 1.0) if peak > 0 else 0.0
        if dd < mdd:
            mdd, mdd_start, mdd_end = dd, ts_vals[peak_idx], ts_vals[i]

    realized = [float(t["realized_pnl"]) for t in trades if "realized_pnl" in t]
    if realized:
        wins = [x for x in realized if x > 0]
        losses = [-x for x in realized if x < 0]
        win_rate = (len(wins) / len(realized)) if realized else None
        payoff = (sum(wins) / len(wins)) / (sum(losses) / len(losses)) if wins and losses else None
        profit_factor = (sum(wins) / sum(losses)) if wins and losses and sum(losses) > 0 else None
    else:
        win_rate = payoff = profit_factor = None

    return {
        "initial_equity": initial,
        "final_equity": final,
        "total_return": total_ret,
        "mdd": mdd,
        "mdd_start_ts": mdd_start,
        "mdd_end_ts": mdd_end,
        "trade_count": len(trades),
        "win_rate": win_rate,
        "payoff": payoff,
        "profit_factor": profit_factor,
    }


# ---------- 메타 빌더 ----------

def build_run_meta(
    *,
    mode: Optional[str] = None,
    base_currency: str,
    open_ts_utc: str,
    plan: Optional[Mapping[str, Any]] = None,
    broker_conf: Optional[Mapping[str, Any]] = None,
    run_type: Optional[str] = None,  # 과거 호환
    policy: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """실행 메타 생성(호출부 시그니처와 정합)."""
    run_mode = mode or run_type or "live"
    meta: dict[str, Any] = {
        "schema_version": "1.0",
        "mode": run_mode,
        "base_currency": base_currency,
        "open_ts_utc": open_ts_utc,
        "plan": dict(plan or {}),
        "broker_conf": dict(broker_conf or {}),
        "created_utc": _utc_now_iso(),
    }
    if policy:
        meta["policy"] = dict(policy)
        for k in ("two_stage", "alloc_mode", "reserve_pct", "downsize_retries", "price_cushion_pct", "fee_cushion_pct"):
            if k in policy:
                meta[k] = policy[k]
    return meta


# ---------- 파일 기록 ----------

def _write_csv(path: str, rows: list[Mapping[str, object]]) -> None:
    # 헤더: 필수 키 우선 + 합집합
    preferred = ["ts", "symbol", "side", "qty", "price", "commission", "tax", "reason", "equity"]
    keys: list[str] = list(preferred)
    seen = set(preferred)
    for r in rows or []:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows or []:
            w.writerow({k: r.get(k, "") for k in keys})


def _write_json(path: str, obj: Mapping[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_outputs(
    out_dir: str,
    *,
    trades: list[Mapping[str, object]],
    equity_curve: list[Mapping[str, object]],
    metrics: Mapping[str, object],
    run_meta: Mapping[str, object],
    orders_log: Iterable[Mapping[str, object]] | None = None,
    fills_log: Iterable[Mapping[str, object]] | None = None,
    rejected_orders: Iterable[Mapping[str, object]] | None = None,
    cash_flow_summary: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """디스크 기록 후 경로 맵 반환."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "trades_csv": os.path.join(out_dir, "trades.csv"),
        "equity_curve_csv": os.path.join(out_dir, "equity_curve.csv"),
        "metrics_json": os.path.join(out_dir, "metrics.json"),
        "run_meta_json": os.path.join(out_dir, "run_meta.json"),
    }
    _write_csv(paths["trades_csv"], make_trades(trades))
    _write_csv(paths["equity_curve_csv"], make_equity_curve(equity_curve))
    _write_json(paths["metrics_json"], dict(metrics))
    _write_json(paths["run_meta_json"], dict(run_meta))
    if rejected_orders is not None:
        p = os.path.join(out_dir, "rejected_orders.csv")
        _write_csv(p, list(rejected_orders))
        paths["rejected_orders_csv"] = p
    if cash_flow_summary is not None:
        p = os.path.join(out_dir, "cash_flow_summary.json")
        _write_json(p, dict(cash_flow_summary))
        paths["cash_flow_summary_json"] = p
    if orders_log is not None:
        p = os.path.join(out_dir, "orders.jsonl")
        _write_jsonl(p, orders_log)
        paths["orders_jsonl"] = p
    if fills_log is not None:
        p = os.path.join(out_dir, "fills.jsonl")
        _write_jsonl(p, fills_log)
        paths["fills_jsonl"] = p
    return paths


# ---------- 패키징 ----------

def finalize_outputs(
    *,
    out_dir: Optional[str] = None,
    trades: Iterable[Mapping[str, object]],
    equity_curve: Iterable[Mapping[str, object]],
    run_meta: Mapping[str, object],
    metrics: Optional[Mapping[str, object]] = None,
    artifacts: Optional[Mapping[str, object]] = None,
    orders_log: Iterable[Mapping[str, object]] | None = None,
    fills_log: Iterable[Mapping[str, object]] | None = None,
    rejected_orders: Iterable[Mapping[str, object]] | None = None,
    cash_details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """
    산출물 패키징:
    - metrics 미제공 시 내부 계산
    - out_dir 제공 시 파일 기록 후 파일 경로를 artifacts에 병합
    - artifacts 인자가 있으면 보존(orders/fills 목록 등)
    - rejected_orders/cash_details 전달 시 추가 산출물 기록
    """
    tlist = make_trades(trades)
    elist = make_equity_curve(equity_curve)
    mcalc = dict(metrics) if metrics is not None else compute_metrics(elist, tlist)

    ret_artifacts: dict[str, object] = {}
    if artifacts:
        ret_artifacts.update(dict(artifacts))

    orders_src = ret_artifacts.get("orders") if "orders" in ret_artifacts else orders_log
    fills_src = ret_artifacts.get("fills") if "fills" in ret_artifacts else fills_log
    rejected_src = ret_artifacts.get("rejected_orders") if (artifacts and "rejected_orders" in artifacts) else rejected_orders

    cash_flow_summary = None
    if cash_details is not None:
        cds = dict(cash_details)
        denom = float(cds.get("cash_start", 0.0)) + float(cds.get("proceeds_realized", 0.0))
        util = (float(cds.get("cash_used_for_buys", 0.0)) / denom) if denom > 0 else 0.0
        cash_flow_summary = {
            "cash_start": float(cds.get("cash_start", 0.0)),
            "proceeds_realized": float(cds.get("proceeds_realized", 0.0)),
            "cash_used_for_buys": float(cds.get("cash_used_for_buys", 0.0)),
            "cash_end": float(cds.get("cash_end", cds.get("unspent_cash", 0.0))),
            "cash_utilization": max(0.0, min(1.0, util)),
        }

    file_paths: dict[str, str] = {}
    if out_dir:
        file_paths = write_outputs(
            out_dir,
            trades=tlist,
            equity_curve=elist,
            metrics=mcalc,
            run_meta=run_meta,
            orders_log=orders_src if isinstance(orders_src, Iterable) else None,
            fills_log=fills_src if isinstance(fills_src, Iterable) else None,
            rejected_orders=rejected_src if isinstance(rejected_src, Iterable) else None,
            cash_flow_summary=cash_flow_summary,
        )
        ret_artifacts.update(file_paths)

    return {
        "trades": tlist,
        "equity_curve": elist,
        "metrics": mcalc,
        "run_meta": dict(run_meta),
        "artifacts": ret_artifacts,
    }

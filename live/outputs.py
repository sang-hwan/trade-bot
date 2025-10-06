# live/outputs.py
"""
목적:
- 실매매 결과를 시뮬레이션과 동일 스키마로 정리/기록한다.
- 산출물: trades, equity_curve, metrics, run_meta (+선택: 주문/체결 로그)

시간 규약:
- 모든 타임스탬프 문자열은 UTC ISO8601(Z).

파일 출력:
- out_dir가 주어지면 CSV/JSON/JSONL로 기록(표준 라이브러리 우선).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional
import csv
import json
import os

# 시뮬레이션 로직 재사용(가능 시). 실패하면 로컬 구현으로 대체.
try:
    from simulation.outputs import (  # type: ignore
        make_trades_df as _sim_make_trades_df,
        make_equity_curve as _sim_make_equity_curve_df,
        compute_metrics as _sim_compute_metrics,
    )
except Exception:
    _sim_make_trades_df = None
    _sim_make_equity_curve_df = None
    _sim_compute_metrics = None


# ---------- 정규화 유틸 ----------

def _as_float(x) -> float:
    return float(x)

def _ensure_ts(x: str) -> str:
    return str(x).strip()

def _sorted_by_ts(rows: List[dict]) -> List[dict]:
    return sorted(rows, key=lambda r: r.get("ts") or r.get("ts_utc") or "")


# ---------- 공개 API: 빌더들 ----------

def make_trades(trades: Iterable[Mapping[str, object]]) -> List[dict]:
    """
    trades 정규화:
    - 최소 키: ts, symbol, side, qty, price, commission, reason
    - 정렬: ts 오름차순
    - 추가 키(있으면 보존)
    """
    if _sim_make_trades_df is not None:
        df = _sim_make_trades_df(trades)  # pandas.DataFrame(ts 오름차순)
        out = df.to_dict(orient="records")
        norm: List[dict] = []
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

    # ── 로컬 fallback
    out: List[dict] = []
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


def make_equity_curve(points: Iterable[Mapping[str, object]]) -> List[dict]:
    """
    equity_curve 정규화:
    - 각 원소: {"ts": "...Z", "equity": float}
    - 정렬: ts 오름차순
    """
    if _sim_make_equity_curve_df is not None:
        ec = _sim_make_equity_curve_df(points)  # DataFrame(index=ts, col 'equity')
        if getattr(ec, "empty", True):
            return []
        ec_reset = ec.reset_index()
        return _sorted_by_ts([{"ts": _ensure_ts(str(ts)), "equity": _as_float(eq)}
                              for ts, eq in zip(ec_reset["ts"], ec_reset["equity"])])
    # ── 로컬 fallback
    out: List[dict] = []
    for p in points:
        ts = _ensure_ts(p.get("ts", ""))
        eq = _as_float(p["equity"])
        out.append({"ts": ts, "equity": eq})
    return _sorted_by_ts(out)


def compute_metrics(equity_curve: List[Mapping[str, object]], trades: List[Mapping[str, object]]) -> Dict[str, object]:
    """
    기본 지표:
    - 시뮬레이션 결과 우선(total_return/mdd/거래통계), 보조 필드(initial/final/trade_count) 보강
    - 시뮬 모듈 부재 시 로컬 계산 사용
    """
    if _sim_compute_metrics is not None and _sim_make_equity_curve_df is not None and _sim_make_trades_df is not None:
        ec_df = _sim_make_equity_curve_df(equity_curve)
        initial_equity = float(ec_df["equity"].iloc[0]) if (getattr(ec_df, "empty", True) is False) else 0.0
        trades_df = _sim_make_trades_df(trades)
        sim_m = _sim_compute_metrics(ec_df, initial_equity=initial_equity, trades_df=trades_df)
        final_equity = float(ec_df["equity"].iloc[-1]) if (getattr(ec_df, "empty", True) is False) else 0.0
        return {
            "initial_equity": initial_equity,
            "final_equity": final_equity,
            "total_return": float(sim_m.get("total_return", 0.0)),
            "mdd": float(sim_m.get("mdd", 0.0)),
            "mdd_start_ts": None,
            "mdd_end_ts": None,
            "trade_count": int(sim_m.get("trades", 0)),
            "win_rate": sim_m.get("win_rate"),
            "payoff": sim_m.get("payoff"),
            "profit_factor": sim_m.get("profit_factor"),
        }

    # ── 로컬 fallback
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


def build_run_meta(
    *,
    engine_params: Mapping[str, object] | None = None,
    price_columns_used: Mapping[str, str] | None = None,
    snapshot_meta: Mapping[str, object] | None = None,
    broker_meta: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
    equity_curve: Iterable[Mapping[str, object]] | None = None,
) -> Dict[str, object]:
    """run_meta 표준화: 공통 필드 구성, 제공 메타 얕은 병합. start/end는 equity_curve 기준."""
    meta: Dict[str, object] = {"schema_version": "1.0", "mode": "live"}
    if engine_params:
        meta["engine_params"] = dict(engine_params)
    if price_columns_used:
        meta["price_columns_used"] = dict(price_columns_used)
    if snapshot_meta:
        meta["snapshot_meta"] = dict(snapshot_meta)
    if broker_meta:
        meta["broker_meta"] = dict(broker_meta)
    if extra:
        meta.update(dict(extra))
    if equity_curve:
        ecs = make_equity_curve(equity_curve)
        if ecs:
            meta["start_ts"] = ecs[0]["ts"]
            meta["end_ts"] = ecs[-1]["ts"]
    return meta


# ---------- 파일 기록 ----------

def _write_csv(path: str, rows: List[Mapping[str, object]]) -> None:
    # 헤더는 모든 키의 합집합(필수 키 우선); 데이터가 없어도 헤더 기록
    preferred = ["ts", "symbol", "side", "qty", "price", "commission", "tax", "reason", "equity"]
    keys: List[str] = list(preferred)
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
    trades: List[Mapping[str, object]],
    equity_curve: List[Mapping[str, object]],
    metrics: Mapping[str, object],
    run_meta: Mapping[str, object],
    orders_log: Iterable[Mapping[str, object]] | None = None,
    fills_log: Iterable[Mapping[str, object]] | None = None,
) -> Dict[str, str]:
    """
    디스크에 산출물을 기록하고 경로 맵을 반환.
    - trades.csv / equity_curve.csv / metrics.json / run_meta.json
    - orders.jsonl / fills.jsonl (옵션)
    """
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

    if orders_log is not None:
        p = os.path.join(out_dir, "orders.jsonl")
        _write_jsonl(p, orders_log)
        paths["orders_jsonl"] = p
    if fills_log is not None:
        p = os.path.join(out_dir, "fills.jsonl")
        _write_jsonl(p, fills_log)
        paths["fills_jsonl"] = p
    return paths


def finalize_outputs(
    trades: Iterable[Mapping[str, object]],
    equity_curve: Iterable[Mapping[str, object]],
    run_meta: Mapping[str, object],
    *,
    out_dir: Optional[str] = None,
    orders_log: Iterable[Mapping[str, object]] | None = None,
    fills_log: Iterable[Mapping[str, object]] | None = None,
) -> Dict[str, object]:
    """
    산출물 패키징:
    - metrics는 equity_curve와 trades에서 계산
    - out_dir가 있으면 파일 기록 후 경로(artifacts) 포함
    """
    tlist = make_trades(trades)
    elist = make_equity_curve(equity_curve)
    metrics = compute_metrics(elist, tlist)
    artifacts: Dict[str, str] = {}
    if out_dir:
        artifacts = write_outputs(
            out_dir,
            trades=tlist,
            equity_curve=elist,
            metrics=metrics,
            run_meta=run_meta,
            orders_log=orders_log,
            fills_log=fills_log,
        )
    return {
        "trades": tlist,
        "equity_curve": elist,
        "metrics": metrics,
        "run_meta": dict(run_meta),
        "artifacts": artifacts,
    }

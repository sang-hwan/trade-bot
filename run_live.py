# run_live.py
"""실매매 실행 스크립트 (Python 3.11+), AUTO 전용.
판단: D-1 종가(UTC) / 집행: D 시가(UTC)
우선순위: Stop → Signal → Rebalance
산출물: trades / equity_curve / metrics / run_meta (+ orders/fills 로그)

요점
- live/broker_hub.py로 브로커 빌드/라우팅/예산스케일/집행 공통화.
- KIS(국내·해외) + Upbit 동시 지원.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional
from datetime import datetime, timezone
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


# ── 소형 유틸 ────────────────────────────────────────────────────────────────

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rows.append(json.loads(s))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return rows


def _append_jsonl(path: str, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── 환경(.env → os.environ) ─────────────────────────────────────────────────

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

def _load_env_file(path: str) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                if k and (k not in os.environ):
                    os.environ[k] = v.strip()
    except FileNotFoundError:
        return


def _expand_env(value):
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


# ── RUNS_ROOT 보장 ───────────────────────────────────────────────────────────
def _ensure_runs_root() -> Path:
    """
    RUNS_ROOT 디렉터리를 실행 시점에 보장.
    - 환경변수 RUNS_ROOT가 있으면 그 경로 사용
    - 없으면 <프로젝트 루트>/runs 사용
    - 존재하지 않으면 생성(parents=True)
    """
    root_env = os.environ.get("RUNS_ROOT", "").strip()
    base = Path(__file__).resolve().parent
    path = Path(root_env).expanduser().resolve() if root_env else (base / "runs")
    if not root_env:
        os.environ["RUNS_ROOT"] = str(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── 메인 ────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    """계획(AUTO) → 주문(Stop→Signal→Rebalance) →(옵션) 루프 폴링/대사/산출."""
    p = argparse.ArgumentParser(description="실매매 실행 스크립트 (AUTO 전용)")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)

    # 계획 산출
    p.add_argument("--source", required=True)
    p.add_argument("--interval", required=True)
    p.add_argument("--plan-out", default=None)

    # 체결 옵션
    p.add_argument("--fills", default=None)
    p.add_argument("--collect-fills", action="store_true")
    p.add_argument("--market", default=None)  # NASD/NYSE…
    p.add_argument("--collect-seconds", type=int, default=10)
    p.add_argument("--collect-params", default="{}")
    p.add_argument("--fills-out", default=None)

    # 루프 옵션
    p.add_argument("--loop", action="store_true")
    p.add_argument("--poll-fills-every", type=int, default=5)
    p.add_argument("--reconcile-every", type=int, default=30)
    p.add_argument("--max-runtime", type=int, default=0)

    args = p.parse_args(argv)

    # RUNS_ROOT 보장(스크립트 의존 제거)
    _ensure_runs_root()

    # 설정 로드
    _load_env_file(".env")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_raw = json.load(f)
    cfg = _expand_env(cfg_raw)

    # 미해결 ${ENV} 검사
    def _scan_env_placeholders(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, str) and _ENV_VAR_PATTERN.search(v):
                    raise SystemExit(f"[config] unresolved env var in {k}: {v}")
                _scan_env_placeholders(v)
        elif isinstance(d, list):
            for v in d:
                _scan_env_placeholders(v)
    _scan_env_placeholders(cfg)

    # 로컬 모듈
    try:
        from live.plan_auto import build_plan_auto
        from live.order_router import RouterInputs, build_orders
        from live.reconcile import reconcile
        from live.outputs import finalize_outputs, build_run_meta
        from live.fills import collect_fills_loop
        from live.broker_hub import (
            build_brokers,
            route_broker,
            fetch_price_map,
            get_cash,
            normalize_positions_kis,
            scale_buys_by_budget,
            send_orders_serial,
        )
        from live.broker_adapter import OrderRequest as KisOrderRequest  # type: ignore
        from live.broker_adapter_upbit import OrderRequest as UpbitOrderRequest  # type: ignore
    except ImportError as e:
        print(f"[fatal] module import failed: {e}", file=sys.stderr)
        raise

    # 공통 파라미터
    base_ccy: str = cfg.get("base_currency", "KRW")
    lot_step = float(cfg.get("lot_step", 1.0))
    lot_step_upbit = float(cfg.get("lot_step_upbit", 0.0001))
    price_step = float(cfg.get("price_step", 0.0))
    open_ts = cfg.get("open_ts_utc", _utc_now_iso())
    tif = str(cfg.get("tif", "OPG"))

    # 계획 산출
    stops_list, signals_map, rebal_spec = build_plan_auto(cfg, source=args.source, interval=args.interval)

    if args.plan_out:
        os.makedirs(args.plan_out, exist_ok=True)
        with open(os.path.join(args.plan_out, "stops.json"), "w", encoding="utf-8") as f:
            json.dump(stops_list, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.plan_out, "signals.json"), "w", encoding="utf-8") as f:
            json.dump(signals_map, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.plan_out, "rebal.json"), "w", encoding="utf-8") as f:
            json.dump(rebal_spec, f, ensure_ascii=False, indent=2)

    # 대상 집합
    stops: list[str] = list(stops_list)
    signals: dict[str, float] = {k: float(v) for k, v in signals_map.items() if v is not None}
    g_syms = list((rebal_spec.get("sell_notional") or {}).keys()) if isinstance(rebal_spec, dict) else []
    l_syms = list((rebal_spec.get("buy_notional") or {}).keys()) if isinstance(rebal_spec, dict) else []
    symbols_union = set(signals.keys()) | set(stops) | set(g_syms) | set(l_syms)
    if not symbols_union:
        print("[info] no target symbols for today; exiting.")
        return 0

    # 브로커 빌드
    brokers = build_brokers(cfg)
    kis = brokers.get("KIS")
    upbit = brokers.get("UPBIT")

    # 시세 맵
    price_map = fetch_price_map(sorted(symbols_union), brokers, retries=2)

    # 현금/포지션 스냅샷
    cash_kis = get_cash("KIS", kis, cfg)
    cash_up = get_cash("UPBIT", upbit, cfg)

    positions_kis: dict[str, float] = {}
    if kis:
        try:
            pos_resp = kis.fetch_positions()  # type: ignore[union-attr]
            rows = (pos_resp.get("output1") or pos_resp.get("output") or []) if isinstance(pos_resp, dict) else []
            positions_kis = normalize_positions_kis(rows if isinstance(rows, list) else [])
        except Exception:
            positions_kis = {}

    # 주문 생성
    rinp = RouterInputs(
        positions=positions_kis,
        stops=sorted(stops),
        signals=signals,
        rebal_spec=rebal_spec,
        price_map=price_map,
        lot_step=lot_step,
        price_step=price_step,
        open_ts_utc=open_ts,
        tif=tif,
    )
    orders_all: list[dict[str, Any]] = build_orders(rinp)

    # 브로커별 그룹
    kis_orders: list[dict[str, Any]] = []
    upbit_orders: list[dict[str, Any]] = []
    for o in orders_all:
        (upbit_orders if route_broker(o["symbol"]) == "UPBIT" else kis_orders).append(o)

    # Stage 1: 매도(선집행)
    sent: list[dict[str, Any]] = []
    sent += send_orders_serial("KIS", kis, KisOrderRequest, kis_orders, side="sell", lot_step=lot_step, downsize_retries=int(cfg.get("downsize_retries", 2)))
    sent += send_orders_serial("UPBIT", upbit, UpbitOrderRequest, upbit_orders, side="sell", lot_step=lot_step_upbit, downsize_retries=int(cfg.get("downsize_retries", 2)))

    # (옵션) KIS 체결 폴링
    fills_rows: list[dict[str, Any]] = []
    if args.collect_fills and kis:
        oc = (cfg.get("broker_kis") or cfg.get("broker") or {}).get("overseas_conf", {}) if args.market else {}
        base_params = oc.get("fills_params", {}) if isinstance(oc.get("fills_params", {}), dict) else {}
        try:
            user_params = json.loads(args.collect_params or "{}")
        except (json.JSONDecodeError, TypeError):
            user_params = {}
        collect_params = dict(base_params); collect_params.update(user_params)
        stage1 = collect_fills_loop(
            broker=kis,  # type: ignore[arg-type]
            fills_path=(oc.get("fills_path") if isinstance(oc, dict) else None),
            tr_fills=((oc.get("tr", {}) or {}).get("fills") if isinstance(oc, dict) else None),
            market=args.market,
            seconds=int(args.collect_seconds),
            extra_params=collect_params,
        ) or []
        fills_rows.extend(stage1)
        if args.fills_out and stage1:
            _append_jsonl(args.fills_out, stage1)

    # Stage 2: 매수(브로커별 예산 스케일)
    reserve_pct = float(cfg.get("reserve_pct", 0.01))
    price_cushion_pct = float(cfg.get("price_cushion_pct", 0.001))
    fee_cushion_pct = float(cfg.get("fee_cushion_pct", 0.0005))
    budget_kis = max(0.0, get_cash("KIS", kis, cfg) * (1.0 - reserve_pct))
    budget_up = max(0.0, get_cash("UPBIT", upbit, cfg) * (1.0 - reserve_pct))
    upbit_bands = ((cfg.get("broker_upbit") or cfg.get("upbit") or {}).get("tick_bands")) or None

    kis_buys = scale_buys_by_budget(
        "KIS", kis_orders, budget=budget_kis, lot_step=lot_step, price_map=price_map,
        price_cushion_pct=price_cushion_pct, fee_cushion_pct=fee_cushion_pct,
    )
    up_buys = scale_buys_by_budget(
        "UPBIT", upbit_orders, budget=budget_up, lot_step=lot_step_upbit, price_map=price_map,
        price_cushion_pct=price_cushion_pct, fee_cushion_pct=fee_cushion_pct, upbit_tick_bands=upbit_bands,
    )

    sent += send_orders_serial("KIS", kis, KisOrderRequest, kis_buys, side="buy", lot_step=lot_step, downsize_retries=int(cfg.get("downsize_retries", 2)))
    sent += send_orders_serial("UPBIT", upbit, UpbitOrderRequest, up_buys, side="buy", lot_step=lot_step_upbit, downsize_retries=int(cfg.get("downsize_retries", 2)))

    # 이후: 체결 수집 → 대사 → 산출물
    cash_start = float(cash_kis) + float(cash_up)
    pos_value_kis = sum(float(positions_kis.get(sym, 0.0)) * float(price_map.get(sym, 0.0)) for sym in positions_kis)
    equity0 = cash_start + pos_value_kis

    os.makedirs(args.out, exist_ok=True)
    run_meta = build_run_meta(
        mode="live",
        base_currency=base_ccy,
        open_ts_utc=open_ts,
        plan={
            "stops_count": len(stops),
            "signals_count": len([k for k, v in signals.items() if v > 0]),
            "rebal_has_targets": bool(g_syms or l_syms),
        },
        broker_conf={
            "kis_use_paper": bool((cfg.get("broker_kis") or cfg.get("broker") or {}).get("use_paper", False)),
            "has_upbit": bool(upbit is not None),
        },
    )
    run_meta.update({
        "two_stage": True,
        "reserve_pct": reserve_pct,
        "alloc_mode": "proportional_per_broker",
        "downsize_retries": int(cfg.get("downsize_retries", 2)),
        "price_cushion_pct": price_cushion_pct,
        "fee_cushion_pct": fee_cushion_pct,
        "initial_equity": float(equity0),
    })

    def _make_fee_fn(commission_rate: float, tax_rate_sell: float):
        def fee(notional: float, side: str, symbol: str):
            notional_abs = abs(float(notional))
            commission = notional_abs * float(commission_rate)
            tax = notional_abs * float(tax_rate_sell) if side == "sell" else 0.0
            return commission, tax
        return fee

    def _reconcile_and_write(all_fills: list[dict[str, Any]]):
        fee_fn = _make_fee_fn(
            float(cfg.get("commission_rate", 0.0)),
            float(cfg.get("tax_rate", 0.0) or cfg.get("fee", {}).get("tax_rate_sell", 0.0)),
        )
        reco = reconcile(
            intended_orders=orders_all,
            fills=all_fills,
            starting_positions=positions_kis,
            starting_cash={base_ccy: cash_start},
            base_currency=base_ccy,
            fee_fn=fee_fn,
        )
        cash_base = float(reco.cash.get(base_ccy) or reco.cash.get("cash_end") or 0.0)
        pos_val_now = 0.0
        for sym, q in (reco.positions or {}).items():
            try:
                pos_val_now += float(q) * float(price_map.get(sym, 0.0))
            except (TypeError, ValueError):
                continue
        equity_curve = [
            {"ts": open_ts, "equity": float(run_meta["initial_equity"])},
            {"ts": _utc_now_iso(), "equity": cash_base + pos_val_now},
        ]
        packed = finalize_outputs(
            out_dir=args.out,
            trades=reco.trades,
            equity_curve=equity_curve,
            run_meta=run_meta,
            artifacts={"orders": sent, "fills": all_fills},
        )
        return packed, reco

    # 누적 fills 버퍼
    all_fills: list[dict[str, Any]] = list(fills_rows or [])

    def _collect_append(seconds: int):
        nonlocal all_fills
        if args.fills:
            if os.path.exists(args.fills):
                all_fills = _read_jsonl(args.fills)
            return
        if not args.collect_fills or not kis:
            return
        if not args.market:
            raise SystemExit("--market is required when --collect-fills is on")
        oc = (cfg.get("broker_kis") or cfg.get("broker") or {}).get("overseas_conf", {}) or {}
        try:
            user_params = json.loads(args.collect_params or "{}")
        except (json.JSONDecodeError, TypeError):
            user_params = {}
        base_params = oc.get("fills_params", {}) if isinstance(oc.get("fills_params", {}), dict) else {}
        collect_params = dict(base_params); collect_params.update(user_params)
        new_rows = collect_fills_loop(
            broker=kis,  # type: ignore[arg-type]
            fills_path=oc.get("fills_path"),
            tr_fills=(oc.get("tr", {}) or {}).get("fills"),
            market=args.market,
            seconds=max(0, int(seconds)),
            extra_params=collect_params,
        ) or []
        if new_rows:
            all_fills.extend(new_rows)
            if args.fills_out:
                _append_jsonl(args.fills_out, new_rows)

    # 실행 모드
    if not args.loop:
        _collect_append(int(args.collect_seconds))
        packed, reco = _reconcile_and_write(all_fills)
        print(json.dumps({
            "artifacts": packed.get("artifacts", {}),
            "metrics": packed.get("metrics", {}),
            "positions": reco.positions,
            "cash": reco.cash,
            "unmatched_orders": reco.unmatched_orders,
            "unmatched_fills": reco.unmatched_fills,
        }, ensure_ascii=False, indent=2))
        return 0

    start = time.monotonic()
    next_poll = start
    next_rec = start
    _reconcile_and_write(all_fills)

    while True:
        now = time.monotonic()
        if args.max_runtime and (now - start) >= float(args.max_runtime):
            break
        if now >= next_poll:
            _collect_append(int(args.poll_fills_every))
            next_poll += float(max(1, args.poll_fills_every))
        if now >= next_rec:
            _reconcile_and_write(all_fills)
            next_rec += float(max(1, args.reconcile_every))
        time.sleep(1)

    packed, reco = _reconcile_and_write(all_fills)
    print(json.dumps({
        "artifacts": packed.get("artifacts", {}),
        "metrics": packed.get("metrics", {}),
        "positions": reco.positions,
        "cash": reco.cash,
        "unmatched_orders": reco.unmatched_orders,
        "unmatched_fills": reco.unmatched_fills,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

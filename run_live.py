# run_live.py
"""
실매매 실행 스크립트 (Python 3.11+), AUTO 전용
- 계획 산출: D-1 종가(UTC)까지만 사용(룩어헤드 금지)
- 집행 우선순위: Stop → Signal → Rebalance
- 산출물: trades / equity_curve / metrics / run_meta (+ orders/fills 로그)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional
from datetime import datetime, timezone
import argparse
import json
import os
import sys

# 로컬 모듈
try:
    from live.broker_adapter import KisBrokerAdapter, OrderRequest, BrokerError
    from live.order_router import RouterInputs, build_orders
    from live.reconcile import reconcile
    from live.outputs import finalize_outputs, build_run_meta
    from live.fills import collect_fills_loop           # 체결 폴링 수집
    from live.plan_auto import build_plan_auto          # AUTO 계획 산출
except ImportError as e:
    print(f"[fatal] module import failed: {e}", file=sys.stderr)
    raise


# ---------- 공통 I/O 유틸 ----------

def _read_json(path: Optional[str], default: Any) -> Any:
    """JSON 로드: path 미지정 시 default 반환."""
    if not path:
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """JSONL → list[dict]. 공백 라인은 무시."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out

def _append_jsonl(path: str, rows: Iterable[Mapping[str, Any]]) -> None:
    """rows를 JSONL 파일에 추가. 상위 디렉터리가 있으면 생성."""
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(dict(r), ensure_ascii=False) + "\n")

def _utc_now_iso() -> str:
    """UTC ISO-8601("...Z") 문자열."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- 수수료 ----------

def _make_fee_fn(commission_rate: float, tax_rate_sell: float):
    """수수료/세금 계산 함수."""
    cr = float(commission_rate or 0.0)
    tr = float(tax_rate_sell or 0.0)

    def fee_fn(notional: float, side: str, symbol: str) -> tuple[float, float]:
        commission = abs(notional) * cr
        tax = abs(notional) * (tr if side == "sell" else 0.0)
        return commission, tax

    return fee_fn


# ---------- 시세/포지션/주문 ----------

def _fetch_price_map(broker: KisBrokerAdapter, symbols: Iterable[str]) -> Dict[str, float]:
    """{symbol: price} 생성."""
    out: Dict[str, float] = {}
    for sym in symbols:
        if not sym:
            continue
        px, _ = broker.fetch_price(sym)
        out[sym] = float(px)
    return out

def _parse_positions_krx(resp: Mapping[str, Any]) -> Dict[str, float]:
    """KIS 잔고 응답 → {symbol: qty}."""
    pos: Dict[str, float] = {}
    rows = resp.get("output1") or resp.get("output") or []
    if not isinstance(rows, list):
        return pos
    for r in rows:
        code = (r.get("pdno") or r.get("PDNO") or r.get("symbol") or "")
        code = str(code).strip()
        if not code:
            continue
        qty_raw = r.get("hldg_qty") or r.get("HLDG_QTY") or r.get("qty") or 0
        try:
            qty = float(qty_raw)
        except (TypeError, ValueError):
            continue
        if qty == 0:
            continue
        symbol = code if ":" in code else f"KRX:{code}"
        pos[symbol] = pos.get(symbol, 0.0) + qty
    return pos

def _place_orders(broker: KisBrokerAdapter, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """라우터 산출 주문을 브로커로 전송."""
    results: List[Dict[str, Any]] = []
    for o in orders:
        req = OrderRequest(symbol=o["symbol"], side=o["side"], qty=int(o["qty"]), price=o.get("price"), market=None)
        res = broker.place_order(req)
        results.append({
            **o,
            "broker": "KIS",
            "order_id": res.order_id,
            "ok": bool(res.ok),
            "raw": res.raw,
            "ts_sent": _utc_now_iso(),
        })
    return results


# ---------- 에쿼티 ----------

def _equity_after(cash_ccy: float, positions: Mapping[str, float], price_map: Mapping[str, float]) -> float:
    """현금+보유자산 시가 평가 합."""
    eq = float(cash_ccy)
    for sym, q in positions.items():
        eq += float(q) * float(price_map.get(sym, 0.0))
    return float(eq)


# ---------- 메인 ----------

def main(argv: Optional[List[str]] = None) -> int:
    """계획(AUTO) → 라우팅/주문 → 대사 → 산출물."""
    p = argparse.ArgumentParser(description="실매매 실행 스크립트 (AUTO 전용)")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)

    # AUTO 옵션
    p.add_argument("--source", default="yahoo")
    p.add_argument("--interval", default="1d")
    p.add_argument("--plan-out", default=None, help="계획(stops/signals/rebal) 저장 디렉터리")

    # 체결 옵션
    p.add_argument("--fills", default=None, help="체결 JSONL(외부 공급)")
    p.add_argument("--collect-fills", action="store_true", help="KIS API 단기 폴링 수집")
    p.add_argument("--market", default=None, help="NASD/NYSE 등 (collect 시 필수)")
    p.add_argument("--collect-seconds", type=int, default=10)
    p.add_argument("--collect-params", default="{}")
    p.add_argument("--fills-out", default=None)

    args = p.parse_args(argv)

    # 설정
    cfg = _read_json(args.config, {})
    base_ccy: str = cfg.get("base_currency", "KRW")
    lot_step = cfg.get("lot_step", 1.0)
    price_step = cfg.get("price_step", 0.0)
    open_ts = cfg.get("open_ts_utc", _utc_now_iso())
    tif = cfg.get("tif", "OPG")

    # 계획 산출(AUTO)
    stops_list, signals_map, rebal_spec = build_plan_auto(cfg, source=args.source, interval=args.interval)
    if args.plan_out:
        os.makedirs(args.plan_out, exist_ok=True)
        with open(os.path.join(args.plan_out, "stops.json"), "w", encoding="utf-8") as f:
            json.dump(stops_list, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.plan_out, "signals.json"), "w", encoding="utf-8") as f:
            json.dump(signals_map, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.plan_out, "rebal.json"), "w", encoding="utf-8") as f:
            json.dump(rebal_spec, f, ensure_ascii=False, indent=2)
    stops = set(stops_list)
    signals = signals_map

    # 브로커
    bro_conf = cfg.get("broker", {})
    broker = KisBrokerAdapter(
        app_key=bro_conf["app_key"],
        app_secret=bro_conf["app_secret"],
        cano=bro_conf["cano"],
        acnt_prdt_cd=bro_conf.get("acnt_prdt_cd", "01"),
        use_paper=bool(bro_conf.get("use_paper", False)),
        overseas_conf=bro_conf.get("overseas_conf", None),
    )

    # 시세
    symbols_union = set(stops) | set(signals.keys()) \
        | set(rebal_spec.get("buy_notional", {}).keys()) \
        | set(rebal_spec.get("sell_notional", {}).keys())
    price_map = _fetch_price_map(broker, sorted(symbols_union))

    # 포지션
    try:
        positions = _parse_positions_krx(broker.fetch_positions())
    except BrokerError:
        positions = {}

    starting_cash = {base_ccy: float(cfg.get("initial_cash", 0.0))}

    # 주문 생성(Stop → Signal → Rebalance)
    rinp = RouterInputs(
        positions=positions,
        stops=sorted(stops),
        signals=signals,
        rebal_spec=rebal_spec,
        price_map=price_map,
        lot_step=lot_step,
        price_step=price_step,
        open_ts_utc=open_ts,
        tif=tif,
    )
    orders = build_orders(rinp)

    # 주문 전송
    try:
        orders_log = _place_orders(broker, orders)
    except BrokerError as e:
        print(f"[error] place_orders failed: {e}", file=sys.stderr)
        orders_log = []

    # 체결(우선순위: 수집 → 파일 → 없음)
    fills: List[Dict[str, Any]] = []
    if args.collect_fills:
        ocfg = bro_conf.get("overseas_conf", {}) or {}
        fills_path = ocfg.get("fills_path")
        tr_fills = (ocfg.get("tr") or {}).get("fills")
        if not (fills_path and tr_fills and args.market):
            raise SystemExit("collect-fills 사용 시 fills_path / tr.fills / --market 필요")
        try:
            extra_params = json.loads(args.collect_params)
        except json.JSONDecodeError:
            raise SystemExit("--collect-params 는 유효한 JSON 문자열이어야 합니다.")
        fills = collect_fills_loop(
            broker,
            fills_path=fills_path,
            tr_fills=tr_fills,
            market=args.market,
            seconds=int(args.collect_seconds),
            extra_params=extra_params,
        )
        if args.fills_out and fills:
            _append_jsonl(args.fills_out, fills)
    elif args.fills:
        fills = _read_jsonl(args.fills)

    # 대사
    fee_cfg = cfg.get("fee", {})
    fee_fn = _make_fee_fn(fee_cfg.get("commission_rate", 0.0), fee_cfg.get("tax_rate_sell", 0.0))
    reco = reconcile(
        intended_orders=orders,
        fills=fills,
        starting_positions=positions,
        starting_cash=starting_cash,
        base_currency=base_ccy,
        fee_fn=fee_fn,
        qty_tol=0.0,
        price_tol=0.0,
    )

    # 에쿼티(시작/종료 2점)
    start_eq = float(starting_cash.get(base_ccy, 0.0))
    end_px_map = _fetch_price_map(broker, set(reco.positions.keys()) or symbols_union)
    end_eq = _equity_after(reco.cash.get(base_ccy, 0.0), reco.positions, end_px_map)
    equity_curve = [{"ts": open_ts, "equity": start_eq}, {"ts": _utc_now_iso(), "equity": end_eq}]

    # 산출물
    run_meta = build_run_meta(
        engine_params={
            "tif": tif,
            "base_currency": base_ccy,
            "auto_mode": True,
            "data_source": args.source,
            "interval": args.interval,
        },
        price_columns_used={"open": "open_adj|open", "close": "close_adj|close"},
        snapshot_meta=None,
        broker_meta={"broker": "KIS", "paper": broker.use_paper},
        extra={"note": "live run"},
        equity_curve=equity_curve,
    )
    packed = finalize_outputs(
        trades=reco.trades,
        equity_curve=equity_curve,
        run_meta=run_meta,
        out_dir=args.out,
        orders_log=orders_log,
        fills_log=fills,
    )

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

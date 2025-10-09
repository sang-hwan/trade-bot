# run_live.py
"""실매매 실행 스크립트 (Python 3.11+), AUTO 전용.
판단: D-1 종가(UTC) / 집행: D 시가(UTC)
우선순위: Stop → Signal → Rebalance
산출물: trades / equity_curve / metrics / run_meta (+ orders/fills 로그)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
from datetime import datetime, timezone
import argparse
import json
import os
import re
import sys
from dataclasses import asdict, is_dataclass

# 최소 버전 가드
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ required.")

# 로컬 모듈
try:
    from live.broker_adapter import KisBrokerAdapter, OrderRequest, BrokerError
    from live.order_router import RouterInputs, build_orders
    from live.reconcile import reconcile
    from live.outputs import finalize_outputs, build_run_meta
    from live.fills import collect_fills_loop
    from live.plan_auto import build_plan_auto
except ImportError as e:
    print(f"[fatal] module import failed: {e}", file=sys.stderr)
    raise

# ---------- 환경변수 로드/치환 ----------

def _load_env_file(path: str = ".env") -> None:
    """간단한 .env 로더: KEY=VALUE. 주석/빈줄 무시, OS 값은 보존."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and (os.environ.get(k) is None):
                    os.environ[k] = v
    except FileNotFoundError:
        return

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

def _expand_env(value):
    """${VAR} 패턴을 os.environ 값으로 재귀 치환."""
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value

def _to_jsonable(obj: Any) -> Any:
    """객체를 JSON 가능 형태로 변환."""
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return str(obj)

# ---------- I/O ----------

def _read_json(path: Optional[str], default: Any) -> Any:
    """JSON 로드 (path 없으면 default)."""
    if not path:
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """JSONL→list[dict] (공백 라인 무시)."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out

def _append_jsonl(path: str, rows: List[Mapping[str, Any]]) -> None:
    """rows를 JSONL에 append (디렉터리 자동 생성)."""
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(dict(r), ensure_ascii=False) + "\n")

def _utc_now_iso() -> str:
    """UTC ISO-8601("...Z")."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- 수수료 ----------

def _make_fee_fn(commission_rate: float, tax_rate_sell: float = 0.0):
    """reconcile 호환: fee(notional, side, symbol) -> (commission, tax)."""
    def fee(notional: float, side: str, symbol: str):
        gross = abs(float(notional))
        commission = gross * float(commission_rate)
        tax = gross * float(tax_rate_sell) if str(side).lower() == "sell" else 0.0
        return commission, tax
    return fee

# ---------- 브로커 포맷 파서 ----------

def _parse_positions_krx(rows: List[Mapping[str, Any]]) -> Dict[str, float]:
    """브로커 포맷 → {symbol: qty}. 비 dict 입력은 건너뜀."""
    out: Dict[str, float] = {}
    for r in rows:
        if not isinstance(r, Mapping):
            continue
        sym = str(r.get("symbol") or r.get("pdno") or "").strip()
        try:
            qty = float(r.get("qty") or r.get("hldg_qty") or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        if sym:
            out[sym] = out.get(sym, 0.0) + qty
    return out

# ---------- 메인 ----------

def main(argv: Optional[List[str]] = None) -> int:
    """계획(AUTO) → 라우팅/주문 → 대사 → 산출물."""
    p = argparse.ArgumentParser(description="실매매 실행 스크립트 (AUTO 전용)")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)

    # 계획 산출(AUTO)
    p.add_argument("--source", required=True, help="데이터 소스")
    p.add_argument("--interval", required=True, help="1d/1h 등")
    p.add_argument("--plan-out", default=None, help="계획 저장 디렉터리")

    # 체결 옵션
    p.add_argument("--fills", default=None, help="체결 JSONL(외부 공급)")
    p.add_argument("--collect-fills", action="store_true", help="KIS API 단기 폴링 수집")
    p.add_argument("--market", default=None, help="NASD/NYSE 등 (collect 시 필수)")
    p.add_argument("--collect-seconds", type=int, default=10)
    p.add_argument("--collect-params", default="{}")
    p.add_argument("--fills-out", default=None)

    args = p.parse_args(argv)

    # 설정(.env → config → env 치환)
    _load_env_file(".env")
    if not os.path.exists(args.config):
        raise SystemExit(f"[config] file not found: {args.config}")
    try:
        cfg = _expand_env(_read_json(args.config, {}))
    except Exception as e:
        raise SystemExit(f"[config] failed to load/parse JSON: {args.config} — {e}")

    # 필수 브로커 키 치환 확인
    bro_conf = cfg.get("broker", {}) or {}
    for _k in ("app_key", "app_secret", "cano", "acnt_prdt_cd"):
        _val = str(bro_conf.get(_k, ""))
        if _val.startswith("${") and _val.endswith("}"):
            raise SystemExit(f"[config] environment variable not expanded: {_val} (check .env/OS env).")

    base_ccy: str = cfg.get("base_currency", "KRW")
    lot_step = cfg.get("lot_step", 1.0)
    price_step = cfg.get("price_step", 0.0)
    open_ts = cfg.get("open_ts_utc", _utc_now_iso())
    tif = cfg.get("tif", "OPG")

    # 계획 산출(AUTO)
    stops_list, signals_map, rebal_spec = build_plan_auto(cfg, source=args.source, interval=args.interval)

    # 계획 저장(옵션)
    if args.plan_out:
        os.makedirs(args.plan_out, exist_ok=True)
        with open(os.path.join(args.plan_out, "stops.json"), "w", encoding="utf-8") as f:
            json.dump(stops_list, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.plan_out, "signals.json"), "w", encoding="utf-8") as f:
            json.dump(signals_map, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.plan_out, "rebalancing_spec.json"), "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(rebal_spec), f, ensure_ascii=False, indent=2)

    # 스탑/신호
    stops: List[str] = list(stops_list)
    signals: Dict[str, int] = {k: int(v) for k, v in signals_map.items() if v is not None}

    # 브로커
    broker = KisBrokerAdapter(
        app_key=bro_conf["app_key"],
        app_secret=bro_conf["app_secret"],
        cano=bro_conf["cano"],
        acnt_prdt_cd=bro_conf.get("acnt_prdt_cd", "01"),
        use_paper=bool(bro_conf.get("use_paper", False)),
        overseas_conf=bro_conf.get("overseas_conf", None),
    )

    # 대상 심볼
    rebal_obj = rebal_spec if isinstance(rebal_spec, Mapping) else {}
    g_syms = list(rebal_obj.get("g_symbols", []) or [])
    l_syms = list(rebal_obj.get("l_symbols", []) or [])
    symbols_union = set(stops) | set(signals.keys()) | set(g_syms) | set(l_syms)
    if not symbols_union:
        print("[info] no target symbols for today; exiting.")
        return 0

    # 시세
    def _fetch_price_map(broker: KisBrokerAdapter, symbols: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for sym in symbols:
            px, _ccy = broker.fetch_price(sym)
            out[sym] = float(px)
        return out

    price_map = _fetch_price_map(broker, sorted(symbols_union))

    # 현금/포지션
    try:
        cash_resp = broker.fetch_cash()
        cash = float(cash_resp.get("cash_ccy") or cash_resp.get("cash") or 0.0)
    except BrokerError:
        cash = float(cfg.get("initial_cash", 0.0))

    try:
        _pos_resp = broker.fetch_positions()
        _rows = (_pos_resp.get("output1") or _pos_resp.get("output") or []) if isinstance(_pos_resp, Mapping) else (_pos_resp or [])
        _rows = _rows if isinstance(_rows, list) else []
        positions = _parse_positions_krx(_rows)
    except BrokerError:
        positions = {}

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
        tif=str(tif),
    )
    _orders: List[Dict[str, Any]] = build_orders(rinp)  # dict 리스트(대사용 원본)

    # 브로커 송신용 OrderRequest 변환
    orders: List[OrderRequest] = [
        OrderRequest(
            symbol=o["symbol"],
            side=o["side"],
            qty=int(o["qty"]),
            price=o.get("price"),
            market=args.market,
        )
        for o in _orders
    ]

    # 주문 전송
    sent: List[Dict[str, Any]] = []
    for o in orders:
        try:
            resp = broker.place_order(o)
            sent.append({"ts": _utc_now_iso(), "symbol": o.symbol, "side": o.side, "qty": o.qty, "tif": rinp.tif, "resp": resp})
        except BrokerError as e:
            sent.append({"ts": _utc_now_iso(), "symbol": o.symbol, "side": o.side, "qty": o.qty, "tif": rinp.tif, "error": str(e)})

    # 체결 수집
    fills_rows: List[Dict[str, Any]] = []
    if args.fills:
        if not os.path.exists(args.fills):
            raise SystemExit(f"[fills] file not found: {args.fills}")
        fills_rows = _read_jsonl(args.fills)
    elif args.collect_fills:
        if not args.market:
            raise SystemExit("--market is required when --collect-fills is on")
        oc = bro_conf.get("overseas_conf", {}) or {}
        # config 기본값(fills_params) + CLI 인자 병합
        user_params = json.loads(args.collect_params or "{}")
        base_params = oc.get("fills_params", {}) if isinstance(oc.get("fills_params", {}), dict) else {}
        collect_params = dict(base_params)
        collect_params.update(user_params)
        fills_rows = collect_fills_loop(
            broker=broker,
            fills_path=oc.get("fills_path"),
            tr_fills=(oc.get("tr", {}) or {}).get("fills"),
            market=args.market,
            seconds=int(args.collect_seconds),
            extra_params=collect_params,
        )
        if args.fills_out and fills_rows:
            _append_jsonl(args.fills_out, fills_rows)

    # 대사(reconcile)
    fee_fn = _make_fee_fn(
        float(cfg.get("commission_rate", 0.0)),
        float(cfg.get("tax_rate", 0.0) or cfg.get("fee", {}).get("tax_rate_sell", 0.0)),
    )
    reco = reconcile(
        intended_orders=_orders,              # 라우터 원본 dict 리스트
        fills=fills_rows,                     # 폴링/외부 공급 체결 dict 리스트
        starting_positions=positions,
        starting_cash={base_ccy: cash},       # 기준통화: 현금
        base_currency=base_ccy,
        fee_fn=fee_fn,
    )

    # 산출물 패키징
    os.makedirs(args.out, exist_ok=True)
    run_meta = build_run_meta(
        mode="live",
        base_currency=base_ccy,
        open_ts_utc=open_ts,
        plan={
            "stops_count": len(stops),
            "signals_count": len(signals),
            "rebal_has_targets": bool(g_syms or l_syms),
        },
        broker_conf={"use_paper": bool(bro_conf.get("use_paper", False))},
    )
    equity_val = float(reco.cash.get(base_ccy, 0.0))
    for sym, qty in (reco.positions or {}).items():
        try:
            px, _ = broker.fetch_price(sym)
        except Exception:
            pxs = [float(f["price"]) for f in fills_rows if str(f.get("symbol")) == sym]
            px = (sum(pxs) / len(pxs)) if pxs else 0.0
        equity_val += float(qty) * float(px)
    equity_rows = [{"ts": open_ts, "equity": equity_val}]

    packed = finalize_outputs(
        out_dir=args.out,
        trades=reco.trades,
        equity_curve=equity_rows,
        run_meta=run_meta,
        artifacts={"orders": sent, "fills": fills_rows},
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

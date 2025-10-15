# run_live.py
"""실매매 실행 스크립트 (Python 3.11+), AUTO 전용.
판단: D-1 종가(UTC) / 집행: D 시가(UTC)
우선순위: Stop → Signal → Rebalance
산출물: trades / equity_curve / metrics / run_meta (+ orders/fills 로그)

변경점
- 주문을 브로커별로 그룹핑(KIS: KRX/NASD/NYSE…, Upbit: KRW-/USDT-/BTC-)하고,
  KisBrokerAdapter와 UpbitBrokerAdapter로 병렬/순차 집행.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional
from datetime import datetime, timezone
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time


# ---------- 유틸 ----------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return rows


def _append_jsonl(path: str, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------- 환경(.env → os.environ) ----------

def _load_env_file(path: str) -> None:
    """KEY=VALUE 형식의 .env 로더(기존 os.environ 값은 유지)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k and (k not in os.environ):
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


# ---------- 심볼 라우팅 ----------

_KIS_PREFIXES = {"KRX", "KOSPI", "KOSDAQ", "NASD", "NASDAQ", "NYSE", "AMEX", "NAS"}
_UPBIT_BASES = {"KRW", "USDT", "BTC"}


def _route_broker_by_symbol(symbol: str) -> str:
    """심볼 접두/형식으로 브로커 라우팅: 'KIS' | 'UPBIT'."""
    s = symbol.strip().upper()
    if ":" in s:
        pref, rest = s.split(":", 1)
        if pref == "UPBIT":
            return "UPBIT"
        if pref in _KIS_PREFIXES:
            return "KIS"
        s = rest
    if "-" in s:
        base = s.split("-", 1)[0]
        if base in _UPBIT_BASES:
            return "UPBIT"
    return "KIS"


# ---------- 메인 ----------

def main(argv: Optional[list[str]] = None) -> int:
    """계획(AUTO) → 주문(Stop→Signal→Rebalance) →(선택)루프 폴링/대사/산출."""
    p = argparse.ArgumentParser(description="실매매 실행 스크립트 (AUTO 전용)")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)

    # 계획 산출(AUTO)
    p.add_argument("--source", required=True, help="데이터 소스")
    p.add_argument("--interval", required=True, help="1d/1h 등")
    p.add_argument("--plan-out", default=None, help="계획 저장 디렉터리")

    # 체결 옵션
    p.add_argument("--fills", default=None, help="체결 JSONL(외부 공급)")
    p.add_argument("--collect-fills", action="store_true", help="KIS API 폴링 수집")
    p.add_argument("--market", default=None, help="NASD/NYSE 등 (collect 시 필수)")
    p.add_argument("--collect-seconds", type=int, default=10)
    p.add_argument("--collect-params", default="{}")
    p.add_argument("--fills-out", default=None)

    # 루프 옵션
    p.add_argument("--loop", action="store_true", help="세션 동안 주기적으로 폴링/대사/산출물 갱신")
    p.add_argument("--poll-fills-every", type=int, default=5, help="체결 폴링 주기(초)")
    p.add_argument("--reconcile-every", type=int, default=30, help="대사/산출물 갱신 주기(초)")
    p.add_argument("--max-runtime", type=int, default=0, help="최대 실행 시간(초). 0이면 제한 없음")

    args = p.parse_args(argv)

    # 설정(.env → config → env 치환)
    _load_env_file(".env")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_raw = json.load(f)
    cfg = _expand_env(cfg_raw)

    # 미해결 ${ENV} 남아 있으면 실패
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
        from live.broker_adapter import KisBrokerAdapter, OrderRequest as KisOrderRequest
        from live.broker_adapter_upbit import UpbitBrokerAdapter, OrderRequest as UpbitOrderRequest
        from live.order_router import RouterInputs, build_orders
        from live.reconcile import reconcile
        from live.outputs import finalize_outputs, build_run_meta
        from live.fills import collect_fills_loop
        from live.plan_auto import build_plan_auto
    except ImportError as e:
        print(f"[fatal] module import failed: {e}", file=sys.stderr)
        raise

    # 환경값
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
        with open(os.path.join(args.plan_out, "rebal.json"), "w", encoding="utf-8") as f:
            json.dump(rebal_spec, f, ensure_ascii=False, indent=2)

    # 스탑/신호
    stops: list[str] = list(stops_list)
    signals: dict[str, float] = {k: float(v) for k, v in signals_map.items() if v is not None}

    # 브로커 구성(KIS + Upbit)
    bro_conf_kis = cfg.get("broker_kis") or cfg.get("broker", {}) or {}
    kis: Optional[KisBrokerAdapter] = None
    if bro_conf_kis:
        kis = KisBrokerAdapter(
            app_key=bro_conf_kis["app_key"],
            app_secret=bro_conf_kis["app_secret"],
            cano=bro_conf_kis["cano"],
            acnt_prdt_cd=bro_conf_kis.get("acnt_prdt_cd", "01"),
            use_paper=bool(bro_conf_kis.get("use_paper", False)),
            overseas_conf=bro_conf_kis.get("overseas_conf", None),
        )

    up_conf = cfg.get("broker_upbit") or cfg.get("upbit") or cfg.get("broker", {}).get("upbit") or {}
    upbit: Optional[UpbitBrokerAdapter] = None
    if up_conf:
        upbit = UpbitBrokerAdapter(
            access_key=up_conf["access_key"],
            secret_key=up_conf["secret_key"],
            tick_bands=up_conf.get("tick_bands"),
        )

    # 대상 심볼 집합
    g_syms = list((rebal_spec.get("sell_notional") or {}).keys()) if isinstance(rebal_spec, dict) else []
    l_syms = list((rebal_spec.get("buy_notional") or {}).keys()) if isinstance(rebal_spec, dict) else []
    symbols_union = set(signals.keys()) | set(stops) | set(g_syms) | set(l_syms)
    if not symbols_union:
        print("[info] no target symbols for today; exiting.")
        return 0

    # 가격 맵(브로커 라우팅)
    def _fetch_price_map(symbols: Iterable[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for sym in symbols:
            broker_tag = _route_broker_by_symbol(sym)
            try:
                if broker_tag == "UPBIT":
                    if not upbit:
                        continue
                    px, _ = upbit.fetch_price(sym)
                else:
                    if not kis:
                        continue
                    px, _ = kis.fetch_price(sym)
                out[sym] = float(px)
            except Exception as e:
                print(f"[WARN] fetch_price failed for {sym}: {e}", file=sys.stderr)
        return out

    price_map = _fetch_price_map(sorted(symbols_union))

    # 현금/포지션(집행 전 스냅샷)
    def _safe_get_cash_kis() -> float:
        if not kis:
            return float(cfg.get("initial_cash_kis", 0.0))
        try:
            cash_resp = kis.fetch_cash()
            return float(cash_resp.get("cash_ccy") or cash_resp.get("cash") or 0.0)
        except Exception:
            return float(cfg.get("initial_cash_kis", 0.0))

    def _safe_get_cash_upbit() -> float:
        if not upbit:
            return float(cfg.get("initial_cash_upbit", 0.0))
        try:
            r = upbit.fetch_cash()
            return float(r.get("KRW") or 0.0)
        except Exception:
            return float(cfg.get("initial_cash_upbit", 0.0))

    cash_kis = _safe_get_cash_kis()
    cash_up = _safe_get_cash_upbit()

    positions_kis: dict[str, float] = {}
    if kis:
        try:
            _pos_resp = kis.fetch_positions()
            _rows = (_pos_resp.get("output1") or _pos_resp.get("output") or []) if isinstance(_pos_resp, dict) else []
            _rows = _rows if isinstance(_rows, list) else []
            positions_kis = _parse_positions_krx(_rows)
        except Exception:
            positions_kis = {}

    # 주문 생성(Stop → Signal → Rebalance)
    rinp = RouterInputs(
        positions=positions_kis,          # Upbit 포지션 합산은 생략(현금 집행 중심)
        stops=sorted(stops),
        signals=signals,
        rebal_spec=rebal_spec,
        price_map=price_map,
        lot_step=lot_step,                # 유니버스에서 심볼별 주입 가능
        price_step=price_step,
        open_ts_utc=open_ts,
        tif=str(tif),
    )
    orders_all: list[dict[str, Any]] = build_orders(rinp)

    # ---------- 브로커별 그룹핑 ----------
    kis_orders: list[dict[str, Any]] = []
    upbit_orders: list[dict[str, Any]] = []
    for o in orders_all:
        (upbit_orders if _route_broker_by_symbol(o["symbol"]) == "UPBIT" else kis_orders).append(o)

    # ---------- Stage 1: 매도(동시 실행) ----------
    sent: list[dict[str, Any]] = []

    def _send_sells_kis(orders: list[dict[str, Any]]) -> None:
        if not kis or not orders:
            return
        for o in orders:
            if str(o.get("side")).lower() != "sell":
                continue
            try:
                req = KisOrderRequest(symbol=o["symbol"], side="sell", qty=int(o["qty"]), price=o.get("price"), market=None)
                resp = kis.place_order(req)
                sent.append({"ts": _utc_now_iso(), "stage": 1, "broker": "KIS", "symbol": req.symbol, "side": "sell", "qty": req.qty, "tif": rinp.tif, "resp": resp})
            except Exception as e:
                sent.append({"ts": _utc_now_iso(), "stage": 1, "broker": "KIS", "symbol": o["symbol"], "side": "sell", "qty": int(o["qty"]), "tif": rinp.tif, "error": str(e)})

    def _send_sells_upbit(orders: list[dict[str, Any]]) -> None:
        if not upbit or not orders:
            return
        for o in orders:
            if str(o.get("side")).lower() != "sell":
                continue
            try:
                req = UpbitOrderRequest(symbol=o["symbol"], side="sell", qty=float(o["qty"]), price=o.get("price"))
                resp = upbit.place_order(req)
                sent.append({"ts": _utc_now_iso(), "stage": 1, "broker": "UPBIT", "symbol": req.symbol, "side": "sell", "qty": req.qty, "tif": rinp.tif, "resp": resp})
            except Exception as e:
                sent.append({"ts": _utc_now_iso(), "stage": 1, "broker": "UPBIT", "symbol": o["symbol"], "side": "sell", "qty": float(o["qty"]), "tif": rinp.tif, "error": str(e)})

    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        ex.submit(_send_sells_kis, kis_orders)
        ex.submit(_send_sells_upbit, upbit_orders)

    # (옵션) KIS 체결 폴링
    fills_rows: list[dict[str, Any]] = []
    if args.collect_fills and kis:
        oc = (bro_conf_kis.get("overseas_conf", {}) if args.market else {})
        base_params = oc.get("fills_params", {}) if isinstance(oc.get("fills_params", {}), dict) else {}
        try:
            user_params = json.loads(args.collect_params or "{}")
        except (json.JSONDecodeError, TypeError):
            user_params = {}
        collect_params = dict(base_params)
        collect_params.update(user_params)
        _stage1_fills = collect_fills_loop(
            broker=kis,
            fills_path=(oc.get("fills_path") if isinstance(oc, dict) else None),
            tr_fills=((oc.get("tr", {}) or {}).get("fills") if isinstance(oc, dict) else None),
            market=args.market,
            seconds=int(args.collect_seconds),
            extra_params=collect_params,
        )
        if _stage1_fills:
            fills_rows.extend(_stage1_fills)
            if args.fills_out:
                _append_jsonl(args.fills_out, _stage1_fills)

    # ---------- Stage 2: 매수(브로커별 예산으로 각각 다운사이즈) ----------
    reserve_pct = 0.01
    price_cushion_pct = 0.001
    fee_cushion_pct = 0.0005
    downsize_retries = 2

    def _scale_buys(orders: list[dict[str, Any]], *, budget: float, lot_step_scalar: float) -> list[dict[str, Any]]:
        total_need = 0.0
        needs: list[tuple[dict[str, Any], float]] = []
        for o in orders:
            if str(o.get("side")).lower() != "buy":
                continue
            pe = float(o.get("price") or price_map.get(o["symbol"], 0.0) * (1.0 + price_cushion_pct))
            if pe <= 0.0:
                continue
            need = pe * float(o["qty"]) * (1.0 + fee_cushion_pct)
            needs.append((o, need))
            total_need += need
        if total_need <= 0.0 or budget <= 0.0:
            return []
        import math
        scale = min(1.0, float(budget) / total_need)
        lot = float(lot_step_scalar or 0.0)
        out: list[dict[str, Any]] = []
        for o, _ in needs:
            qty = float(o["qty"])
            qty_scaled = (math.floor((qty * scale) / lot) * lot) if lot > 0 else qty * scale
            if qty_scaled >= (lot if lot > 0 else 0.0):
                item = dict(o)
                item["qty"] = qty_scaled
                out.append(item)
        return out

    # 예산(브로커별) 재조회
    cash_kis2 = _safe_get_cash_kis()
    budget_kis = max(0.0, cash_kis2 * (1.0 - reserve_pct))
    cash_up2 = _safe_get_cash_upbit()
    budget_up = max(0.0, cash_up2 * (1.0 - reserve_pct))

    kis_buys_scaled = _scale_buys(kis_orders, budget=budget_kis, lot_step_scalar=float(lot_step))
    up_buys_scaled = _scale_buys(upbit_orders, budget=budget_up, lot_step_scalar=float(cfg.get("lot_step_upbit", 0.0001)))

    def _send_buys_kis(orders: list[dict[str, Any]]) -> None:
        if not kis:
            return
        for o in orders:
            if str(o.get("side")).lower() != "buy":
                continue
            tries = 0
            qty = int(max(0, round(float(o["qty"]))))
            while qty > 0:
                try:
                    req = KisOrderRequest(symbol=o["symbol"], side="buy", qty=int(qty), price=o.get("price"))
                    resp = kis.place_order(req)
                    sent.append({"ts": _utc_now_iso(), "stage": 2, "broker": "KIS", "symbol": req.symbol, "side": "buy", "qty": req.qty, "tif": rinp.tif, "resp": resp})
                    break
                except Exception as e:
                    tries += 1
                    new_qty = qty - int(lot_step)
                    if tries <= downsize_retries and new_qty >= int(lot_step):
                        qty = new_qty
                        continue
                    sent.append({"ts": _utc_now_iso(), "stage": 2, "broker": "KIS", "symbol": o["symbol"], "side": "buy", "qty": int(qty), "tif": rinp.tif, "error": str(e)})
                    break

    def _send_buys_upbit(orders: list[dict[str, Any]]) -> None:
        if not upbit:
            return
        lot_up = float(cfg.get("lot_step_upbit", 0.0001))
        for o in orders:
            if str(o.get("side")).lower() != "buy":
                continue
            tries = 0
            qty = max(0.0, float(o["qty"]))
            while qty > 0.0:
                try:
                    req = UpbitOrderRequest(symbol=o["symbol"], side="buy", qty=float(qty), price=o.get("price"))
                    resp = upbit.place_order(req)
                    sent.append({"ts": _utc_now_iso(), "stage": 2, "broker": "UPBIT", "symbol": req.symbol, "side": "buy", "qty": req.qty, "tif": rinp.tif, "resp": resp})
                    break
                except Exception as e:
                    tries += 1
                    new_qty = qty - lot_up
                    if tries <= downsize_retries and new_qty >= lot_up:
                        qty = new_qty
                        continue
                    sent.append({"ts": _utc_now_iso(), "stage": 2, "broker": "UPBIT", "symbol": o["symbol"], "side": "buy", "qty": float(qty), "tif": rinp.tif, "error": str(e)})
                    break

    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        ex.submit(_send_buys_kis, kis_buys_scaled)
        ex.submit(_send_buys_upbit, up_buys_scaled)

    # ---------- 이후: 체결 수집 → 대사 → 산출물 ----------

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
            "kis_use_paper": bool(bro_conf_kis.get("use_paper", False)),
            "has_upbit": bool(upbit is not None),
        },
    )
    run_meta.update({
        "two_stage": True,
        "reserve_pct": 0.01,
        "alloc_mode": "proportional_per_broker",
        "downsize_retries": int(downsize_retries),
        "price_cushion_pct": float(price_cushion_pct),
        "fee_cushion_pct": float(fee_cushion_pct),
        "initial_equity": float(equity0),
    })

    def _make_fee_fn(commission_rate: float, tax_rate_sell: float):
        """브로커 수수료/세금 모델."""
        def fee(notional: float, side: str, symbol: str):
            notional_abs = abs(float(notional))
            commission = notional_abs * float(commission_rate)
            tax = notional_abs * float(tax_rate_sell) if side == "sell" else 0.0
            return commission, tax
        return fee

    def _reconcile_and_write(all_fills: list[dict[str, Any]]):
        """누적 체결로 대사→자본 업데이트→산출물 기록."""
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
        """한 번 수집해 파일 append 및 누적(KIS만)."""
        nonlocal all_fills
        if args.fills:
            if os.path.exists(args.fills):
                all_fills = _read_jsonl(args.fills)
            return
        if not args.collect_fills or not kis:
            return
        if not args.market:
            raise SystemExit("--market is required when --collect-fills is on")
        oc = bro_conf_kis.get("overseas_conf", {}) or {}
        try:
            user_params = json.loads(args.collect_params or "{}")
        except (json.JSONDecodeError, TypeError):
            user_params = {}
        base_params = oc.get("fills_params", {}) if isinstance(oc.get("fills_params", {}), dict) else {}
        collect_params = dict(base_params)
        collect_params.update(user_params)
        new_rows = collect_fills_loop(
            broker=kis,
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

    # 실행 모드: 원샷 vs 루프
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
    packed, reco = _reconcile_and_write(all_fills)

    while True:
        now = time.monotonic()
        if args.max_runtime and (now - start) >= float(args.max_runtime):
            break
        if now >= next_poll:
            _collect_append(int(args.poll_fills_every))
            next_poll += float(max(1, args.poll_fills_every))
        if now >= next_rec:
            packed, reco = _reconcile_and_write(all_fills)
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


# ---------- 보조 ----------

def _parse_positions_krx(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    """KIS 포지션 응답을 symbol→qty 맵으로 단순화."""
    out: dict[str, float] = {}
    for r in rows:
        sym = str(r.get("pdno") or r.get("symbol") or "")
        qty = float(r.get("hldg_qty") or r.get("qty") or 0.0)
        if sym:
            out[sym] = out.get(sym, 0.0) + qty
    return out


if __name__ == "__main__":
    raise SystemExit(main())

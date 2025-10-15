# live/broker_hub.py
"""
브로커 허브(Registry + Routing + 집행) — Python 3.11+

공개 API
- build_brokers(cfg) -> dict[str, object]
- route_broker(symbol) -> "KIS" | "UPBIT"
- normalize_positions_kis(rows) -> dict[str, float]
- fetch_price_map(symbols, brokers, *, retries=2) -> dict[str, float]
- get_cash(tag, broker, cfg) -> float
- scale_buys_by_budget(tag, orders, *, budget, lot_step, price_map,
                       price_cushion_pct, fee_cushion_pct, upbit_tick_bands=None) -> list[dict]
- send_orders_serial(tag, broker, req_cls, orders, *, side, lot_step, downsize_retries) -> list[dict]

주요 규약
- 시간 표기는 상위(run_live.py)에서 UTC ISO-8601(Z).
- 예산 스케일은 가정가(price_map * (1+buffer)) 기준.
- KIS/Upbit 모두 브로커 단위 직렬 전송(토큰/해시키, idempotency 보호).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional
import math
import threading
import time

__all__ = [
    "build_brokers",
    "route_broker",
    "normalize_positions_kis",
    "fetch_price_map",
    "get_cash",
    "scale_buys_by_budget",
    "send_orders_serial",
]

# 선택 의존(존재하면 사용)
try:
    from live.broker_adapter import KisBrokerAdapter, OrderRequest as KisOrderRequest  # type: ignore
except ImportError:  # 선택 모듈
    KisBrokerAdapter = None  # type: ignore[assignment]
    KisOrderRequest = None   # type: ignore[assignment]

try:
    from live.broker_adapter_upbit import UpbitBrokerAdapter, OrderRequest as UpbitOrderRequest  # type: ignore
except ImportError:  # 선택 모듈
    UpbitBrokerAdapter = None  # type: ignore[assignment]
    UpbitOrderRequest = None   # type: ignore[assignment]


# ---------- 공용 유틸 ----------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- 브로커 레지스트리 ----------

_REGISTRY: dict[str, dict[str, Any]] = {
    "KIS": {
        "req_cls": KisOrderRequest,
        "qty_type": int,
        "cash_getter": lambda b: float((b.fetch_cash() or {}).get("cash_ccy")
                                       or (b.fetch_cash() or {}).get("cash") or 0.0),
        "price_getter": lambda b, sym: float(b.fetch_price(sym)[0]),
        "lot_default": 1.0,
        "lock": threading.Lock(),
    },
    "UPBIT": {
        "req_cls": UpbitOrderRequest,
        "qty_type": float,
        "cash_getter": lambda b: float((b.fetch_cash() or {}).get("KRW") or 0.0),
        "price_getter": lambda b, sym: float(b.fetch_price(sym)[0]),
        "lot_default": 0.0001,
        "lock": threading.Lock(),
    },
}


# ---------- 라우팅 ----------

_KIS_PREFIXES = {"KRX", "KOSPI", "KOSDAQ", "NASD", "NASDAQ", "NYSE", "AMEX", "NAS"}
_UPBIT_BASES = {"KRW", "USDT", "BTC"}


def route_broker(symbol: str) -> str:
    """심볼 형식으로 브로커 라우팅: 기본 'KIS'."""
    s = (symbol or "").strip().upper()
    if ":" in s:
        pref, rest = s.split(":", 1)
        if pref == "UPBIT":
            return "UPBIT"
        if pref in _KIS_PREFIXES:
            return "KIS"
        s = rest
    if "-" in s and s.split("-", 1)[0] in _UPBIT_BASES:
        return "UPBIT"
    return "KIS"


# ---------- 브로커 생성 ----------

def build_brokers(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """
    설정에서 인스턴스 구성.
    - KIS: cfg['broker_kis'] or cfg['broker']
    - UPBIT: cfg['broker_upbit'] or cfg['upbit'] or cfg['broker']['upbit']
    """
    out: dict[str, Any] = {}

    kis_conf = (cfg.get("broker_kis") or cfg.get("broker") or {}) if isinstance(cfg, Mapping) else {}
    if KisBrokerAdapter and isinstance(kis_conf, Mapping) and kis_conf.get("app_key"):
        out["KIS"] = KisBrokerAdapter(
            app_key=kis_conf["app_key"],
            app_secret=kis_conf["app_secret"],
            cano=kis_conf["cano"],
            acnt_prdt_cd=kis_conf.get("acnt_prdt_cd", "01"),
            use_paper=bool(kis_conf.get("use_paper", False)),
            overseas_conf=kis_conf.get("overseas_conf"),
        )

    up_conf = (cfg.get("broker_upbit") or cfg.get("upbit") or (cfg.get("broker", {}) or {}).get("upbit") or {}) if isinstance(cfg, Mapping) else {}
    if UpbitBrokerAdapter and isinstance(up_conf, Mapping) and up_conf.get("access_key"):
        out["UPBIT"] = UpbitBrokerAdapter(
            access_key=up_conf["access_key"],
            secret_key=up_conf["secret_key"],
            tick_bands=up_conf.get("tick_bands"),
        )
    return out


# ---------- 포지션 정규화(KIS) ----------

def normalize_positions_kis(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """
    KIS 포지션을 주문 심볼 포맷으로 변환.
    - 해외: {EXCD}:{SYMB} (예: NASD:AAPL)
    - 국내: KRX:{PDNO}
    """
    out: dict[str, float] = {}
    for r in rows or []:
        excd = str(r.get("excd") or r.get("ovrs_excg_cd") or "").strip().upper()
        symb = str(r.get("symb") or r.get("ovrs_pdno") or "").strip().upper()
        if excd and symb:
            try:
                out[f"{excd}:{symb}"] = out.get(f"{excd}:{symb}", 0.0) + float(r.get("qty") or r.get("hldg_qty") or 0.0)
            except (TypeError, ValueError):
                continue
            continue
        pdno = str(r.get("pdno") or "").strip().upper()
        if pdno:
            try:
                out[f"KRX:{pdno}"] = out.get(f"KRX:{pdno}", 0.0) + float(r.get("hldg_qty") or r.get("qty") or 0.0)
            except (TypeError, ValueError):
                continue
    return out


# ---------- 시세/현금 ----------

def fetch_price_map(symbols: Iterable[str], brokers: Mapping[str, Any], *, retries: int = 2) -> dict[str, float]:
    """브로커 라우팅에 따라 시세를 조회해 심볼→가격 맵 반환(실패 시 소거)."""
    out: dict[str, float] = {}
    for sym in symbols or []:
        tag = route_broker(sym)
        b = brokers.get(tag)
        getter = _REGISTRY.get(tag, {}).get("price_getter")
        if not (b and getter):
            continue
        tries = max(0, int(retries))
        while True:
            try:
                px = float(getter(b, sym))
                if px > 0:
                    out[sym] = px
                break
            except Exception:
                if tries <= 0:
                    break
                tries -= 1
                time.sleep(0.2)
    return out


def get_cash(tag: str, broker: Any, _cfg: Mapping[str, Any]) -> float:
    """브로커별 현금 스냅샷을 단일 수치로 반환(KIS: 통화종류 무관, Upbit: KRW)."""
    getter = _REGISTRY.get(tag, {}).get("cash_getter")
    if not (broker and getter):
        return 0.0
    try:
        val = float(getter(broker))
        return val if math.isfinite(val) and val >= 0.0 else 0.0
    except Exception:
        return 0.0


# ---------- 스케일링(예산) ----------

def _upbit_tick_size_krw(price: float, bands: Optional[list[tuple[float, float]]] = None) -> float:
    """
    Upbit KRW 마켓 가격단위. bands가 주어지면 [(threshold, tick)] 내림차순 사용.
    bands 미제공 시 보수 기본치.
    """
    p = float(price or 0.0)
    if p <= 0:
        return 0.0
    default = [
        (2_000_000, 1000.0),
        (1_000_000, 500.0),
        (500_000, 100.0),
        (100_000, 50.0),
        (10_000, 10.0),
        (1_000, 5.0),
        (100, 1.0),
        (10, 0.1),
        (0, 0.01),
    ]
    for th, tick in (bands or default):
        if p >= float(th):
            return float(tick)
    return 0.01


def _round_up_to_step(x: float, step: float) -> float:
    return float(x) if step <= 0 else math.ceil(float(x) / float(step)) * float(step)


def scale_buys_by_budget(
    tag: str,
    orders: Iterable[Mapping[str, Any]],
    *,
    budget: float,
    lot_step: float,
    price_map: Mapping[str, float],
    price_cushion_pct: float,
    fee_cushion_pct: float,
    upbit_tick_bands: Optional[list[tuple[float, float]]] = None,
) -> list[dict]:
    """
    매수 주문을 예산 내로 축소.
    - scale = min(1, budget / Σ need)
    - qty_scaled = floor((qty*scale)/lot_step)*lot_step
    - Upbit(KRW-*)는 price_est를 틱단위 상향 후 5,000원 최소주문 보장.
    """
    bud = float(budget or 0.0)
    lot = float(lot_step or 0.0)
    if bud <= 0.0:
        return []

    needs: list[tuple[Mapping[str, Any], float, float]] = []
    total_need = 0.0
    for o in orders or []:
        if str(o.get("side")).lower() != "buy":
            continue
        sym = str(o.get("symbol", ""))
        px = float(o.get("price") or price_map.get(sym, 0.0))
        if px <= 0:
            continue
        px_est = px * (1.0 + float(price_cushion_pct or 0.0))
        if tag == "UPBIT" and "-" in sym and sym.split("-", 1)[0] == "KRW":
            tick = _upbit_tick_size_krw(px_est, upbit_tick_bands)
            px_est = _round_up_to_step(px_est, tick) if tick > 0 else px_est
        qty = float(o.get("qty") or 0.0)
        need = px_est * qty * (1.0 + float(fee_cushion_pct or 0.0))
        if qty > 0 and need > 0:
            needs.append((o, qty, need))
            total_need += need

    if total_need <= 0.0:
        return []

    scale = min(1.0, bud / total_need)
    out: list[dict] = []
    for o, qty, _ in needs:
        qty_scaled = qty * scale
        if lot > 0:
            qty_scaled = math.floor(qty_scaled / lot) * lot
        if qty_scaled <= 0:
            continue

        if tag == "UPBIT":
            sym = str(o.get("symbol", ""))
            if "-" in sym and sym.split("-", 1)[0] == "KRW":
                px = float(o.get("price") or price_map.get(sym, 0.0))
                if px > 0:
                    px_est = px * (1.0 + float(price_cushion_pct or 0.0))
                    tick = _upbit_tick_size_krw(px_est, upbit_tick_bands)
                    px_est = _round_up_to_step(px_est, tick) if tick > 0 else px_est
                    if px_est * qty_scaled < 5000.0:
                        if (px_est * max(qty_scaled + lot, lot)) < 5000.0:
                            continue
                        qty_scaled = max(qty_scaled, lot)

        item = dict(o)
        item["qty"] = float(qty_scaled)
        out.append(item)
    return out


# ---------- 직렬 전송 ----------

def _coerce_qty(tag: str, qty: float, lot_step: float) -> float | int:
    """브로커별 수량형식(KIS=int / UPBIT=float) 보정."""
    if tag == "KIS":
        lot = max(float(lot_step or 0.0), 1.0)
        return int(max(0, math.floor(float(qty or 0.0) / lot) * lot))
    return float(qty or 0.0)


def _is_insufficient(e: Exception) -> bool:
    """잔고/예산 부족 계열 오류 탐지."""
    msg = (getattr(e, "args", [""]) or [""])[0]
    code = str(getattr(e, "code", "")).upper()
    text = f"{msg}".upper()
    return (code in {"INSUFFICIENT_CASH", "INSUFFICIENT_FUNDS"}) or any(p in text for p in ("INSUFFICIENT", "잔고", "예수금", "NOT ENOUGH", "부족"))


def send_orders_serial(
    tag: str,
    broker: Any,
    req_cls: type,
    orders: Iterable[Mapping[str, Any]],
    *,
    side: str,
    lot_step: float,
    downsize_retries: int,
) -> list[dict]:
    """
    브로커당 직렬 전송.
    - side 필터 적용.
    - 잔고부족 등 거절 시 lot_step 단위 다운사이즈 재시도.
    - 반환: 전송 로그[{ts, broker, symbol, side, qty_before, qty_sent, lot_step, attempts, resp|error, tif?, reason?}]
    """
    logs: list[dict] = []
    if not (broker and req_cls):
        return logs

    lock = _REGISTRY.get(tag, {}).get("lock")
    qty_type = _REGISTRY.get(tag, {}).get("qty_type", float)

    def _send_one(o: Mapping[str, Any]) -> None:
        sym = str(o.get("symbol", ""))
        if str(o.get("side")).lower() != side.lower():
            return
        price = o.get("price")
        reason = o.get("reason")
        tif = o.get("tif")

        qty_in = float(o.get("qty") or 0.0)
        qty = _coerce_qty(tag, qty_in, lot_step)
        attempts = 0

        while True:
            if (isinstance(qty, (int, float)) and float(qty) <= 0.0) or attempts > max(0, int(downsize_retries)):
                logs.append({
                    "ts": _utc_now_iso(),
                    "broker": tag,
                    "symbol": sym,
                    "side": side.lower(),
                    "qty_before": float(qty_in),
                    "qty_sent": float(qty) if isinstance(qty, (int, float)) else 0.0,
                    "lot_step": float(lot_step or 0.0),
                    "attempts": int(attempts),
                    "error": "qty<=0 or retries_exhausted",
                    "tif": tif,
                    "reason": reason,
                })
                break
            try:
                req = req_cls(symbol=sym, side=side.lower(), qty=qty_type(qty), price=price)
                resp = broker.place_order(req)
                logs.append({
                    "ts": _utc_now_iso(),
                    "broker": tag,
                    "symbol": sym,
                    "side": side.lower(),
                    "qty_before": float(qty_in),
                    "qty_sent": float(qty),
                    "lot_step": float(lot_step or 0.0),
                    "attempts": int(attempts),
                    "resp": resp,
                    "tif": tif,
                    "reason": reason,
                })
                break
            except Exception as e:
                attempts += 1
                if _is_insufficient(e):
                    if tag == "KIS":
                        qty = max(0, int(qty) - int(max(1, round(lot_step))))
                    else:
                        qty = max(0.0, float(qty) - float(lot_step or 0.0))
                    continue
                logs.append({
                    "ts": _utc_now_iso(),
                    "broker": tag,
                    "symbol": sym,
                    "side": side.lower(),
                    "qty_before": float(qty_in),
                    "qty_sent": float(qty),
                    "lot_step": float(lot_step or 0.0),
                    "attempts": int(attempts),
                    "error": str(e),
                    "tif": tif,
                    "reason": reason,
                })
                break

    if lock:
        with lock:
            for o in orders or []:
                _send_one(o)
    else:
        for o in orders or []:
            _send_one(o)
    return logs

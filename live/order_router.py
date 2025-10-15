# live/order_router.py
"""
주문 라우터(자산군 불문, 공통 로직)

공개 API
- apply_cash_budget_proportional(budget, buy_orders, lot_step) -> list[dict]
- build_orders(RouterInputs, *, default_reason_label="signal") -> list[dict]

규약
- 스탑 → 신호 → 리밸런싱 순서로 주문 생성·병합·정렬
- 수량 라운딩은 lot_step(심볼별)만 적용
- price_step 라운딩은 상위 계층/브로커에서 수행(여기서는 아이템포턴시 키 생성에만 반영)
- RouterInputs.price_step/lot_step 은 유니버스에서 심볼별 매핑으로 주입
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping
import hashlib
import math


# ── 내부 유틸 ────────────────────────────────────────────────────────────────

def _get_step(step_cfg: float | Mapping[str, float], symbol: str) -> float:
    """심볼별 step 조회(스칼라/맵 지원). 없으면 0.0."""
    if isinstance(step_cfg, Mapping):
        return float(step_cfg.get(symbol, 0.0))
    return float(step_cfg)


def _floor_to_step(x: float, step: float) -> float:
    """x를 step 배수로 내림. step<=0이면 x 그대로."""
    if step <= 0:
        return x
    return math.floor(x / step) * step


def _make_idemp_key(*parts: str) -> str:
    """입력 동일 → 키 동일."""
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def _merge_or_net(orders: list[dict]) -> list[dict]:
    """같은 심볼의 반대 방향은 상쇄, 같은 방향은 병합(수량 기준)."""
    by_sym: dict[str, dict[str, float]] = {}
    for o in orders:
        q = float(o["qty"])
        if q <= 0:
            continue
        s = by_sym.setdefault(o["symbol"], {"buy": 0.0, "sell": 0.0})
        s[o["side"]] += q

    out: list[dict] = []
    for sym, acc in by_sym.items():
        b, s = acc["buy"], acc["sell"]
        if b > s:
            out.append({"symbol": sym, "side": "buy", "qty": b - s})
        elif s > b:
            out.append({"symbol": sym, "side": "sell", "qty": s - b})
    return out


def _validate_reason(reason: str) -> str:
    r = reason.lower()
    if r not in {"signal", "stop", "rebalance"}:
        raise ValueError("reason must be one of {'signal','stop','rebalance'}")
    return r


# ── 예산 비례 배분 ───────────────────────────────────────────────────────────

def apply_cash_budget_proportional(budget: float, buy_orders: list[dict], lot_step: float) -> list[dict]:
    """
    비례 배분으로 매수 수량을 예산 내로 축소.
      - scale = min(1, budget / Σ cash_need)
      - qty_scaled = floor((requested_qty * scale) / lot_step) * lot_step
      - qty_scaled < lot_step 이면 제거
    반환 주문에는 'funded_qty', 'downsized_by_cash' 추가.
    """
    budget = float(budget or 0.0)
    lot = float(lot_step or 0.0)
    if budget <= 0.0 or lot <= 0.0:
        return []

    needs: list[tuple[dict, float, float]] = []
    total_need = 0.0
    for o in buy_orders:
        rq = float(o.get("requested_qty", o.get("qty", 0.0)) or 0.0)
        cn = o.get("cash_need")
        if cn is None:
            px = float(o.get("price") or 0.0)
            cn = px * rq if px > 0 else 0.0
        cn = float(cn or 0.0)
        if rq <= 0 or cn <= 0:
            continue
        needs.append((o, rq, cn))
        total_need += cn

    if total_need <= 0.0:
        return []

    scale = min(1.0, budget / total_need)
    out: list[dict] = []
    for o, rq, _ in needs:
        qty_scaled = math.floor((rq * scale) / lot) * lot
        if qty_scaled < lot:
            continue
        item = dict(o)
        item["funded_qty"] = qty_scaled
        item["downsized_by_cash"] = bool(qty_scaled < rq)
        out.append(item)

    out.sort(key=lambda x: str(x.get("symbol", "")))  # 결정성 보장
    return out


# ── 공개 API ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RouterInputs:
    positions: Mapping[str, float]                       # 현재 보유 수량
    stops: Iterable[str]                                 # 스탑 대상 심볼 집합/시퀀스
    signals: Mapping[str, float | None]                  # 진입 목표 수량(주); None/<=0 무시
    rebal_spec: Mapping[str, Mapping[str, float]]        # {"buy_notional": {...}, "sell_notional": {...}}
    price_map: Mapping[str, float]                       # 리밸 금액→수량 변환용 가격(예상 시가)
    lot_step: float | Mapping[str, float]                # 유니버스에서 심볼별 주입(수량 라운딩에 사용)
    price_step: float | Mapping[str, float] = 0.0        # 유니버스에서 심볼별 주입(키 생성에만 사용)
    open_ts_utc: str = "1970-01-01T00:00:00Z"
    tif: str = "OPG"                                     # 시장가 at open


def build_orders(inp: RouterInputs, *, default_reason_label: str = "signal") -> list[dict]:
    """
    스탑 → 신호 → 리밸런싱 순서로 주문 생성 후 병합/정렬하여 반환.
    - Stop > Signal 우선순위
    - lot_step 적용(실행 불가 소량 제거)
    - 동일 심볼 상쇄/병합
    - price_step 라운딩은 수행하지 않음(브로커/상위 계층 처리)
    """
    reason_signal = _validate_reason(default_reason_label)
    positions = dict(inp.positions)
    stop_set = set(inp.stops)
    price_map = dict(inp.price_map)

    # 1) STOP — 보유 수량 전량 매도(lot_step 내림)
    orders_stop: list[dict] = []
    for sym in sorted(stop_set):
        pos = float(positions.get(sym, 0.0))
        if pos <= 0:
            continue
        lot = _get_step(inp.lot_step, sym)
        qty = _floor_to_step(pos, lot)
        if qty <= 0:
            continue
        orders_stop.append({"symbol": sym, "side": "sell", "qty": qty, "price": None, "tif": inp.tif, "reason": "stop"})

    # 2) SIGNAL — Stop 대상 제외, 보유 0일 때만 진입(추가매수는 리밸런싱 위임)
    orders_signal: list[dict] = []
    for sym, want in sorted(inp.signals.items()):
        if sym in stop_set or (want is None) or (want <= 0) or positions.get(sym, 0.0) > 0:
            continue
        lot = _get_step(inp.lot_step, sym)
        qty = _floor_to_step(float(want), lot)
        if qty <= 0:
            continue
        orders_signal.append({"symbol": sym, "side": "buy", "qty": qty, "price": None, "tif": inp.tif, "reason": reason_signal})

    # 3) REBAL — 금액→수량, 매도는 보유 한도로 캡
    orders_rebal: list[dict] = []
    buy_notional = dict(inp.rebal_spec.get("buy_notional", {}))
    sell_notional = dict(inp.rebal_spec.get("sell_notional", {}))

    for sym, amt in sorted(buy_notional.items()):
        if amt <= 0:
            continue
        px = float(price_map.get(sym, 0.0))
        if px <= 0:
            continue
        lot = _get_step(inp.lot_step, sym)
        qty = _floor_to_step(amt / px, lot)
        if qty > 0:
            orders_rebal.append({"symbol": sym, "side": "buy", "qty": qty, "price": None, "tif": inp.tif, "reason": "rebalance"})

    for sym, amt in sorted(sell_notional.items()):
        if amt <= 0:
            continue
        px = float(price_map.get(sym, 0.0))
        pos = float(positions.get(sym, 0.0))
        if px <= 0 or pos <= 0:
            continue
        lot = _get_step(inp.lot_step, sym)
        qty = _floor_to_step(min(pos, amt / px), lot)
        if qty > 0:
            orders_rebal.append({"symbol": sym, "side": "sell", "qty": qty, "price": None, "tif": inp.tif, "reason": "rebalance"})

    # 4) 병합/상쇄
    merged = _merge_or_net([*orders_stop, *orders_signal, *orders_rebal])

    # 5) 우선순위 정렬: Stop → Signal → Rebalance, 동일 우선순위 내 심볼 사전순
    priority = {"stop": 0, reason_signal: 1, "rebalance": 2}
    stop_syms = stop_set
    signal_syms = set(inp.signals.keys())

    def _prio(o: dict) -> tuple[int, str]:
        sym = o["symbol"]
        if sym in stop_syms:
            r = "stop"
        elif sym in signal_syms:
            r = reason_signal
        else:
            r = "rebalance"
        o["reason"] = r
        return (priority[r], sym)

    merged.sort(key=_prio)

    # 매도 제한 가드: sell 주문은 stop/rebalance 원천만 허용
    for _o in merged:
        if str(_o.get("side")) == "sell" and str(_o.get("reason")) not in {"stop", "rebalance"}:
            raise ValueError("invalid sell reason: expected {'stop','rebalance'} but got %r" % _o.get("reason"))

    # 6) 아이템포턴시 키(라운딩 규칙 변경 시에도 동일 입력이면 동일 키)
    for o in merged:
        lot = _get_step(inp.lot_step, o["symbol"])
        pstep = _get_step(inp.price_step, o["symbol"])
        o["idempotency_key"] = _make_idemp_key(
            o["symbol"], o["side"], o["reason"], inp.open_ts_utc, f"{lot}", f"{pstep}", f"{o['qty']}"
        )

    return merged

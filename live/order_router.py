# live/order_router.py
"""
주문 라우터:
- 스탑 → 신호 → 리밸런싱 순으로 주문 생성·정렬
- lot_step 규칙 적용, 중복/과도 주문 상쇄·병합
- 최종 주문 리스트를 반환(시장가/OPG 전제; price_step 라운딩은 상위 계층에서 수행)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union
import math
import hashlib

Side = str        # "buy" | "sell"
TIF = str         # "OPG" | "MKT" | "GTC"


# ---------- 내부 유틸 ----------

def _get_step(step_cfg: Union[float, Mapping[str, float]], symbol: str) -> float:
    """심볼별 step 조회(스칼라/맵 지원)."""
    if isinstance(step_cfg, dict):
        return float(step_cfg.get(symbol, 0.0))
    return float(step_cfg)

def _floor_to_step(x: float, step: float) -> float:
    """x를 step 배수로 내림. step<=0이면 x 그대로."""
    if step <= 0:
        return x
    return math.floor(x / step) * step

def _make_idemp_key(*parts: str) -> str:
    """입력 동일 → 키 동일을 보장하는 아이템포턴시 키."""
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]

def _merge_or_net(orders: List[dict]) -> List[dict]:
    """
    같은 심볼의 반대 방향은 상쇄, 같은 방향은 병합(수량 기준).
    """
    by_sym: Dict[str, Dict[str, float]] = {}
    for o in orders:
        q = float(o["qty"])
        if q <= 0:
            continue
        s = by_sym.setdefault(o["symbol"], {"buy": 0.0, "sell": 0.0})
        s[o["side"]] += q

    out: List[dict] = []
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


# ---------- 공개 API ----------

@dataclass(frozen=True)
class RouterInputs:
    positions: Mapping[str, float]                       # 현재 보유 수량
    stops: Iterable[str]                                 # 스탑 대상 심볼 집합/시퀀스
    signals: Mapping[str, Optional[float]]               # 진입 목표 수량(주); None/<=0 무시
    rebal_spec: Mapping[str, Mapping[str, float]]        # {"buy_notional": {...}, "sell_notional": {...}}
    price_map: Mapping[str, float]                       # 리밸 금액→수량 변환용 가격(예상 시가)
    lot_step: Union[float, Mapping[str, float]]
    price_step: Union[float, Mapping[str, float]] = 0.0  # 지정가 정책 사용 시 상위 계층에서 활용
    open_ts_utc: str = "1970-01-01T00:00:00Z"
    tif: TIF = "OPG"                                     # 기본 시장가 at open

def build_orders(inp: RouterInputs, *, default_reason_label: str = "signal") -> List[dict]:
    """
    스탑 → 신호 → 리밸런싱 순서로 주문 생성 후 병합/정렬하여 반환.
    - Stop > Signal 우선순위
    - lot_step 적용(실행 불가 소량 제거)
    - 동일 심볼 상쇄/병합
    """
    reason_signal = _validate_reason(default_reason_label)
    positions = dict(inp.positions)
    stop_set = set(inp.stops)
    price_map = dict(inp.price_map)

    # 1) STOP — 보유 수량 전량 매도(lot_step 내림)
    orders_stop: List[dict] = []
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
    orders_signal: List[dict] = []
    for sym, want in sorted(inp.signals.items()):
        if sym in stop_set or (want is None) or (want <= 0) or positions.get(sym, 0.0) > 0:
            continue
        lot = _get_step(inp.lot_step, sym)
        qty = _floor_to_step(float(want), lot)
        if qty <= 0:
            continue
        orders_signal.append({"symbol": sym, "side": "buy", "qty": qty, "price": None, "tif": inp.tif, "reason": reason_signal})

    # 3) REBAL — 금액→수량, 매도는 보유 한도로 캡
    orders_rebal: List[dict] = []
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

    # 5) 우선순위 정렬: Stop → Signal → Rebalance
    priority = {"stop": 0, reason_signal: 1, "rebalance": 2}
    stop_syms = stop_set
    signal_syms = set(inp.signals.keys())

    def _prio(o: dict) -> Tuple[int, str]:
        sym = o["symbol"]
        if sym in stop_syms:
            r = "stop"
        elif sym in signal_syms:
            r = reason_signal
        else:
            r = "rebalance"
        o["reason"] = r  # reason 보정
        return (priority[r], sym)

    merged.sort(key=_prio)

    # 6) 아이템포턴시 키
    for o in merged:
        lot = _get_step(inp.lot_step, o["symbol"])
        pstep = _get_step(inp.price_step, o["symbol"])
        o["idempotency_key"] = _make_idemp_key(o["symbol"], o["side"], o["reason"], inp.open_ts_utc, f"{lot}", f"{pstep}", f"{o['qty']}")

    return merged

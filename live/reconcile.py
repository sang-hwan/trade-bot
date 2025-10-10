# live/reconcile.py
"""
체결 대사:
- 브로커 체결을 의도 주문과 매칭하고, 수수료/세금 반영 후 포지션·현금 잔액을 갱신.
- 거래 레코드(trades) 스키마는 시뮬레이션과 호환(필수: ts, symbol, side, qty, price, commission, reason).
- 모든 시간 표기는 UTC ISO8601(Z).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional
from datetime import datetime, timezone
import math

_ALLOWED_REASON = {"signal", "stop", "rebalance"}


# ---------- 공개 결과 컨테이너 ----------

@dataclass(slots=True)
class ReconcileOutputs:
    trades: list[dict[str, object]]
    positions: dict[str, float]
    cash: dict[str, float]
    unmatched_orders: list[dict[str, object]]
    unmatched_fills: list[dict[str, object]]


# ---------- 내부 유틸 ----------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _as_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError) as e:
        raise ValueError(f"numeric coercion failed for {x!r}") from e

def _norm_reason(maybe: Optional[str], fallback: str) -> str:
    r = (maybe or fallback or "rebalance").lower()
    return r if r in _ALLOWED_REASON else "rebalance"


def _match(
    intended: list[dict[str, object]],
    fills: list[dict[str, object]],
    *,
    qty_tol: float,
    price_tol: float,
) -> tuple[
    list[tuple[dict[str, object], list[dict[str, object]]]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """
    (symbol, side) 단위로 수량 매칭.
    - 수량 합이 의도 수량에 근접(qty_tol)이면 매칭.
    - intended에 지정가(price)가 있고 price_tol>0이면 가중평균 체결가와의 차이를 기록.
    """
    # intended를 (symbol, side) 하나로 집계(라우터가 병합했더라도 방어적 누적)
    by_key_int: dict[tuple[str, str], dict[str, object]] = {}
    for o in intended:
        k = (str(o["symbol"]), str(o["side"]).lower())
        if k in by_key_int:
            by_key_int[k]["qty"] = _as_float(by_key_int[k]["qty"]) + _as_float(o["qty"])
        else:
            by_key_int[k] = dict(o)

    # fills를 그룹화
    by_key_fill: dict[tuple[str, str], list[dict[str, object]]] = {}
    for f in fills:
        k = (str(f["symbol"]), str(f["side"]).lower())
        by_key_fill.setdefault(k, []).append(f)

    matched: list[tuple[dict[str, object], list[dict[str, object]]]] = []
    unmatched_orders: list[dict[str, object]] = []
    unmatched_fills: list[dict[str, object]] = []

    for k, o in by_key_int.items():
        fills_k = by_key_fill.pop(k, [])
        want = _as_float(o["qty"])
        got = sum(_as_float(f["qty"]) for f in fills_k)

        # 기본: 수량 매칭
        if math.isfinite(want) and abs(got - want) <= max(qty_tol, 0.0):
            matched.append((o, fills_k))
        elif got > 0:
            matched.append((o, fills_k))
            if got < want - max(qty_tol, 0.0):
                rest = dict(o)
                rest["qty_unfilled"] = want - got
                unmatched_orders.append(rest)
            elif got > want + max(qty_tol, 0.0):
                extra = {"symbol": k[0], "side": k[1], "qty_extra": got - want}
                unmatched_fills.append(extra)
        else:
            unmatched_orders.append(dict(o))
            continue

        # 지정가 허용오차(가중평균 체결가 기준) 기록
        try:
            intended_px = o.get("price", None)
            if intended_px is not None and price_tol > 0 and fills_k:
                intended_px = _as_float(intended_px)
                notional = sum(_as_float(f["qty"]) * _as_float(f["price"]) for f in fills_k)
                wavg_px = notional / max(got, 1e-12)
                if abs(wavg_px - intended_px) > price_tol:
                    note = dict(o)
                    note["price_mismatch"] = float(wavg_px - intended_px)
                    unmatched_orders.append(note)
        except (KeyError, ValueError):
            # 의도/체결에 price가 없거나 수치 변환 불가 → 가격 검증 스킵
            pass

    # 남은 키는 의도 없는 체결
    for _, fills_left in by_key_fill.items():
        unmatched_fills.extend(dict(f) for f in fills_left)

    return matched, unmatched_orders, unmatched_fills


# ---------- 공개 함수 ----------

def reconcile(
    intended_orders: Iterable[Mapping[str, object]],
    fills: Iterable[Mapping[str, object]],
    *,
    starting_positions: Mapping[str, float],
    starting_cash: Mapping[str, float],
    base_currency: str,
    fee_fn: Optional[Callable[[float, str, str], tuple[float, float]]] = None,
    qty_tol: float = 1e-9,
    price_tol: float = 0.0,
) -> ReconcileOutputs:
    """
    의도 주문 ↔ 체결을 매칭하고 회계를 갱신.
    - fee_fn(notional, side, symbol) -> (commission_add, tax).
    - 현금은 기준통화 단일 계정으로 갱신(다중 통화는 상위에서 환산).
    - price_tol>0: intended price가 있을 때 가중평균 체결가와의 차이를 확인·기록.
    """
    fee_fn = fee_fn or (lambda notional, side, symbol: (0.0, 0.0))

    # 입력 정규화 & 필수 키 검사
    intended_list: list[dict[str, object]] = [dict(x) for x in intended_orders]
    fills_list: list[dict[str, object]] = [dict(x) for x in fills]

    for o in intended_list:
        for k in ("symbol", "side", "qty"):
            if k not in o:
                raise ValueError(f"intended order missing key: {k}")
    for f in fills_list:
        for k in ("symbol", "side", "qty", "price"):
            if k not in f:
                raise ValueError(f"fill missing key: {k}")

    # 매칭
    matched, unmatched_orders, unmatched_fills = _match(
        intended_list, fills_list, qty_tol=qty_tol, price_tol=price_tol
    )

    # 회계 초기화
    positions: dict[str, float] = {str(k): float(v) for k, v in starting_positions.items()}
    cash: dict[str, float] = {str(k): float(v) for k, v in starting_cash.items()}
    if base_currency not in cash:
        cash[base_currency] = 0.0

    cash_start = float(cash.get(base_currency, 0.0))
    proceeds_realized = 0.0
    cash_used_for_buys = 0.0

    trades: list[dict[str, object]] = []

    # 체결 반영(수수료/세금 포함)
    for order, fills_k in matched:
        sym = str(order["symbol"])
        side = str(order["side"]).lower()
        reason = _norm_reason(order.get("reason"), "rebalance")

        # 주문 대비 실현 요약
        requested_qty = _as_float(order.get("qty", 0.0))
        funded_qty = sum(_as_float(f.get("qty", 0.0)) for f in fills_k)
        rejected_reason = ""
        if funded_qty + max(qty_tol, 0.0) < requested_qty:
            # 부분/무체결. 상위에서 내려온 사유가 있으면 사용
            rejected_reason = str(order.get("rejected_reason") or order.get("reject_reason") or order.get("error_code") or "")

        for f in fills_k:
            qty = _as_float(f["qty"])
            price = _as_float(f["price"])
            ts = str(f.get("ts_utc") or _utc_now_iso())

            commission_broker = _as_float(f.get("commission", 0.0))
            notional = price * qty
            commission_add, tax = fee_fn(abs(notional), side, sym)
            commission_total = commission_broker + float(commission_add)

            if side == "buy":
                positions[sym] = positions.get(sym, 0.0) + qty
                cash_delta = -(notional + commission_total + tax)
                cash[base_currency] = cash.get(base_currency, 0.0) + cash_delta
                cash_used_for_buys += -cash_delta
            elif side == "sell":
                positions[sym] = positions.get(sym, 0.0) - qty
                cash_delta = (notional - commission_total - tax)
                cash[base_currency] = cash.get(base_currency, 0.0) + cash_delta
                proceeds_realized += cash_delta
            else:
                raise ValueError("side must be 'buy' or 'sell'")

            trades.append({
                "ts": ts,
                "symbol": sym,
                "side": side,
                "qty": qty,
                "price": price,
                "commission": commission_total,
                "tax": tax,
                "reason": _norm_reason(f.get("reason"), reason),
                "requested_qty": requested_qty,
                "funded_qty": funded_qty,
                "rejected_reason": rejected_reason,
            })

    cash_end = float(cash.get(base_currency, 0.0))
    cash["cash_start"] = cash_start
    cash["proceeds_realized"] = proceeds_realized
    cash["cash_used_for_buys"] = cash_used_for_buys
    cash["cash_end"] = cash_end
    cash["unspent_cash"] = cash_end

    # unmatched 주문에도 요약 필드 보조(있으면)
    for u in unmatched_orders:
        # 항목 단위로 안전 처리(수치 변환 불가 시 0으로 간주)
        try:
            rq = _as_float(u.get("qty", 0.0))
        except (TypeError, ValueError):
            rq = 0.0
        try:
            unfilled = _as_float(u.get("qty_unfilled", rq))
        except (TypeError, ValueError):
            unfilled = 0.0
        u["requested_qty"] = rq
        u["funded_qty"] = max(rq - unfilled, 0.0)
        if (not u.get("rejected_reason")) and (unfilled > 0.0):
            u["rejected_reason"] = "UNFILLED"

    return ReconcileOutputs(
        trades=trades,
        positions=positions,
        cash=cash,
        unmatched_orders=unmatched_orders,
        unmatched_fills=unmatched_fills,
    )

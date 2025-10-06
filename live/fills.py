# live/fills.py
"""
KIS 체결 수집 헬퍼 (Python 3.11+)
공개 API(핵심)
- collect_fills_loop(broker, *, fills_path, tr_fills, market, seconds, extra_params=None) -> list[dict]
  지정 초(seconds) 동안 폴링하여 체결 리스트를 UTC ISO-8601("...Z") 타임스탬프로 반환.

타이밍/규약
- 체결 타임스탬프는 브로커 응답의 (exec_dt, exec_tm) 조합을 사용하고, 파싱 실패 시 현재 UTC를 사용.
- 중복 방지: 동일(ts_utc|symbol|side|qty|price) 레코드는 1회만 포함.

예외 처리 근거
- 브로커 호출/파싱 시 BrokerError는 폴링 루프 내에서 경고만 남기고 계속 진행(단일 실패로 중단 방지).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import sys
import time

from live.broker_adapter import KisBrokerAdapter, BrokerError


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _first(*vals):
    for v in vals:
        if v not in (None, ""):
            return v
    return None


def _map_fill_row(row: Dict[str, Any], *, market: str, default_reason: str = "signal") -> Optional[Dict[str, Any]]:
    """KIS 체결 응답 → 표준 fills 레코드(dict)로 매핑."""
    code = _first(row.get("PDNO"), row.get("pdno"), row.get("symbol"), row.get("issue_code"))
    if not code:
        return None
    symbol = f"{market}:{code}" if ":" not in str(code) else str(code)

    qty = _first(row.get("ovrs_exec_qty"), row.get("exec_qty"), row.get("qty"))
    price = _first(row.get("ovrs_exec_pric"), row.get("exec_price"), row.get("price"))
    try:
        qty_f = float(qty)
        price_f = float(price)
    except (TypeError, ValueError):
        return None

    side_raw = _first(row.get("side"), row.get("sll_buy_dvsn_cd"), row.get("ord_dvsn"))
    side_map = {"01": "buy", "02": "sell", "B": "buy", "S": "sell"}
    side = side_map.get(str(side_raw).upper(), str(side_raw).lower())
    if side not in ("buy", "sell"):
        return None

    commission = _first(row.get("commission"), row.get("cmssn_amt"), 0.0)
    try:
        commission_f = float(commission or 0.0)
    except (TypeError, ValueError):
        commission_f = 0.0

    dt = _first(row.get("exec_dt"), row.get("ord_dt"))
    tm = _first(row.get("exec_tm"), row.get("ord_tmd"))
    if dt and tm:
        try:
            ts = (
                datetime.strptime(str(dt) + str(tm).zfill(6), "%Y%m%d%H%M%S")
                .replace(tzinfo=timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        except (TypeError, ValueError):
            ts = _utc_now_iso()
    else:
        ts = _utc_now_iso()

    return {
        "ts_utc": ts,
        "symbol": symbol,
        "side": side,
        "qty": qty_f,
        "price": price_f,
        "commission": commission_f,
        "reason": default_reason,
    }


def _pick_tr_id(tr_fills: Any, market: str) -> Optional[str]:
    if isinstance(tr_fills, str):
        return tr_fills
    if isinstance(tr_fills, dict):
        return tr_fills.get(market) or tr_fills.get("default")
    return None


def _fetch_fills_once(
    broker: KisBrokerAdapter,
    *,
    fills_path: str,
    tr_fills: Any,
    market: str,
    extra_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """브로커에서 체결 1회 조회."""
    broker._ensure_token()
    tr_id = _pick_tr_id(tr_fills, market)
    if not tr_id:
        raise BrokerError("TR_ID for fills is missing")
    headers = broker._headers(tr_id, need_auth=True)  # type: ignore[attr-defined]
    params = {"OVRS_EXCG_CD": market}
    if extra_params:
        params.update(extra_params)
    data = broker._request_json("GET", fills_path, headers, params=params)  # type: ignore[attr-defined]
    rows = data.get("output") or data.get("output1") or data.get("results") or []
    if not isinstance(rows, list):
        rows = []
    out: List[Dict[str, Any]] = []
    for r in rows:
        m = _map_fill_row(r, market=market)
        if m:
            out.append(m)
    return out


def collect_fills_loop(
    broker: KisBrokerAdapter,
    *,
    fills_path: str,
    tr_fills: Any,
    market: str,
    seconds: int,
    extra_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    지정 초(seconds) 동안 폴링하여 체결을 수집.
    - BrokerError 발생 시 경고만 출력하고 계속 시도(중단 없음).
    - 중복 레코드는 제거.
    """
    if seconds <= 0:
        return []

    deadline = time.time() + int(seconds)
    acc: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _key(r: Dict[str, Any]) -> str:
        return f"{r['ts_utc']}|{r['symbol']}|{r['side']}|{r['qty']}|{r['price']}"

    while True:
        try:
            rows = _fetch_fills_once(
                broker,
                fills_path=fills_path,
                tr_fills=tr_fills,
                market=market,
                extra_params=extra_params,
            )
            for r in sorted(rows, key=lambda x: x["ts_utc"]):
                k = _key(r)
                if k not in seen:
                    seen.add(k)
                    acc.append(r)
        except BrokerError as e:
            print(f"[warn] fills poll error: {e}", file=sys.stderr)

        if time.time() >= deadline:
            break
        time.sleep(1.0)

    return acc

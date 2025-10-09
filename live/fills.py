# live/fills.py
"""
KIS 체결 수집 헬퍼 (Python 3.11+)

공개 API: collect_fills_loop(...)-> list[dict]
- 지정 초(seconds) 동안 폴링하여 UTC ISO-8601("...Z")로 반환
- 중복 키: ts_utc|symbol|side|qty|price
- 예외: BrokerError는 경고만 출력하고 계속 진행
"""

from __future__ import annotations

from typing import Any
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


def _map_fill_row(
    row: dict[str, Any],
    *,
    market: str,
    default_reason: str = "signal",
) -> dict[str, Any] | None:
    """브로커 체결 응답 → 표준 레코드."""
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


def _pick_tr_id(tr_fills: Any, market: str, *, broker: KisBrokerAdapter | None = None) -> str | None:
    """TR ID 결정: 인자 → broker.overseas_conf.tr.fills."""
    if isinstance(tr_fills, str):
        return tr_fills
    if isinstance(tr_fills, dict):
        return tr_fills.get(market) or tr_fills.get("default")
    if broker is not None:
        conf = getattr(broker, "overseas_conf", {}) or {}
        tr = conf.get("tr", {}) or {}
        val = tr.get("fills")
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            return val.get(market) or val.get("default")
    return None


def _pick_fills_path(fills_path: str | None, *, broker: KisBrokerAdapter | None = None) -> str | None:
    """경로 결정: 인자 → broker.overseas_conf의 흔한 키."""
    if isinstance(fills_path, str) and fills_path:
        return fills_path
    if broker is None:
        return None
    conf = getattr(broker, "overseas_conf", {}) or {}
    for c in (
        conf.get("fills_path"),
        (conf.get("paths", {}) or {}).get("fills"),
        (conf.get("path", {}) or {}).get("fills"),
        (conf.get("endpoints", {}) or {}).get("fills"),
        (conf.get("urls", {}) or {}).get("fills"),
    ):
        if isinstance(c, str) and c:
            return c
    return None


def _market_to_codes(market: str) -> dict[str, str]:
    """시장 문자열 → 국가/시장/거래소 코드."""
    m = (market or "").upper()
    natn_cd = "840"  # USA
    tr_mket_cd_map = {"NASD": "01", "NYSE": "02", "AMEX": "05"}
    excd_map = {"NASD": "NAS", "NYSE": "NYS", "AMEX": "AMS"}
    return {
        "NATN_CD": natn_cd,
        "TR_MKET_CD": tr_mket_cd_map.get(m, "00"),
        "EXCD": excd_map.get(m, ""),
    }


def _fetch_fills_once(
    broker: KisBrokerAdapter,
    *,
    fills_path: str | None = None,
    tr_fills: Any = None,
    market: str,
    extra_params: dict[str, Any] | None = None,
    default_reason: str = "signal",
) -> list[dict[str, Any]]:
    """체결 1회 조회. 실패 시 BrokerError 전파."""
    broker._ensure_token()
    tr_id = _pick_tr_id(tr_fills, market, broker=broker)
    path = _pick_fills_path(fills_path, broker=broker)
    if not tr_id:
        raise BrokerError("TR_ID for fills is missing (pass tr_fills or set broker.overseas_conf.tr.fills)")
    if not path:
        raise BrokerError("fills_path is missing (pass fills_path or set broker.overseas_conf.*.fills)")

    headers = broker._headers(tr_id, need_auth=True)  # type: ignore[attr-defined]

    # 필수 기본 파라미터(계좌/통화/국가/시장/조회구분)
    cano = getattr(broker, "cano", None)
    acnt = getattr(broker, "acnt_prdt_cd", "01")
    if not cano or not acnt:
        raise BrokerError("CANO/ACNT_PRDT_CD is missing on broker adapter")
    mkt = _market_to_codes(market)

    params: dict[str, Any] = {
        "CANO": cano,
        "ACNT_PRDT_CD": acnt,
        "WCRC_FRCR_DVSN_CD": "02",
        "NATN_CD": mkt["NATN_CD"],
        "TR_MKET_CD": mkt["TR_MKET_CD"],
        "INQR_DVSN_CD": "00",
        "EXCD": mkt["EXCD"],
        "OVRS_EXCG_CD": market,
    }
    if extra_params:
        params.update(extra_params)

    data = broker._request_json("GET", path, headers, params=params)  # type: ignore[attr-defined]

    rows = data.get("output") or data.get("output1") or data.get("results") or []
    if not isinstance(rows, list):
        rows = []

    out: list[dict[str, Any]] = []
    for r in rows:
        m = _map_fill_row(r, market=market, default_reason=default_reason)
        if m:
            out.append(m)
    return out


def collect_fills_loop(
    broker: KisBrokerAdapter,
    *,
    fills_path: str | None = None,
    tr_fills: Any = None,
    market: str = "",
    seconds: int = 0,
    extra_params: dict[str, Any] | None = None,
    default_reason: str = "signal",
    **kwargs,
) -> list[dict[str, Any]]:
    """
    seconds 동안 폴링하며 체결 수집.
    - BrokerError: 경고만 출력 후 계속 진행
    - 중복 제거: ts_utc|symbol|side|qty|price
    """
    if seconds <= 0:
        return []

    if extra_params is None and "params" in kwargs:  # 하위호환
        extra_params = kwargs["params"]

    deadline = time.time() + int(seconds)
    acc: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _key(r: dict[str, Any]) -> str:
        return f"{r['ts_utc']}|{r['symbol']}|{r['side']}|{r['qty']}|{r['price']}"

    while True:
        try:
            rows = _fetch_fills_once(
                broker,
                fills_path=fills_path,
                tr_fills=tr_fills,
                market=market,
                extra_params=extra_params,
                default_reason=default_reason,
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

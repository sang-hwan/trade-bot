# live/broker_adapter_upbit.py
"""
Upbit 실매매 어댑터(주문/잔고/시세).

공개 계약
- place_order(OrderRequest) -> OrderResult
- cancel_order(order_id) -> dict
- fetch_positions() -> dict
- fetch_cash() -> dict
- fetch_price(symbol) -> (price: float, raw: dict)

규약
- 인증: JWT(HMAC-SHA256) — Authorization: Bearer <jwt>
  * payload: {"access_key","nonce"[,"query_hash","query_hash_alg":"SHA512"]}
  * query_hash = SHA512(urlencode(params_or_body))
- 레이트리밋: 응답 헤더 'Remaining-Req: group=...; min=...; sec=...' 파싱 → 그룹별 토큰버킷 갱신.
- 최소주문: KRW 마켓 총액 ≥ 5000원 검증.
- 가격단위: tick_bands(가격대별 호가단위)로 지정가 라운딩.

예외 처리
- Upbit 에러(JSON "error": {...}) 및 HTTP 오류는 BrokerError로 승격.
- 표준 라이브러리 우선 사용(urllib, hmac, hashlib 등).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import base64
import hashlib
import hmac
import json
import sys
import time
import uuid
import urllib.parse
import urllib.request
from threading import Lock
from urllib.error import HTTPError, URLError

# Python 3.11+ 요구
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ required.")

API_HOST = "https://api.upbit.com"

__all__ = ["BrokerError", "OrderRequest", "OrderResult", "UpbitBrokerAdapter"]


# ── 공용 타입 ────────────────────────────────────────────────────────────────

class BrokerError(RuntimeError):
    """브로커 API 오류. code/raw/req 포함."""
    def __init__(self, message: str, *, code: Optional[str] = None, raw: Optional[dict] = None, req: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.raw = raw or {}
        self.req = req or {}
    def __str__(self) -> str:  # 간결한 에러 문자열
        c = f"[{self.code}] " if self.code else ""
        return f"{c}{super().__str__()}"


@dataclass(frozen=True)
class OrderRequest:
    """심볼: 'UPBIT:KRW-BTC' | 'KRW-BTC'. side: 'buy' | 'sell'."""
    symbol: str
    side: str
    qty: float
    price: Optional[float] = None  # None: 시장가(매수는 금액 기반이므로 제한)


@dataclass
class OrderResult:
    ok: bool
    order_id: Optional[str]
    raw: dict[str, Any]
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    request: Optional[dict[str, Any]] = None


# ── 레이트리밋(헤더 기반 토큰버킷) ────────────────────────────────────────────

class _RateState:
    """그룹별 남은 쿼터/리셋시각."""
    __slots__ = ("sec_remaining", "min_remaining", "sec_reset_ts", "min_reset_ts")
    def __init__(self) -> None:
        self.sec_remaining = 1
        self.min_remaining = 10
        self.sec_reset_ts = 0.0
        self.min_reset_ts = 0.0


class _RateLimiter:
    """Remaining-Req 헤더(best practice) 기반."""
    def __init__(self) -> None:
        self._lock = Lock()
        self._states: dict[str, _RateState] = {}

    def before(self, group: str) -> None:
        now = time.monotonic()
        with self._lock:
            st = self._states.get(group)
            if st is None:
                return
            if st.sec_remaining <= 0 and now < st.sec_reset_ts:
                time.sleep(max(0.0, st.sec_reset_ts - now) + 0.01)
            if st.min_remaining <= 0 and now < st.min_reset_ts:
                time.sleep(max(0.0, st.min_reset_ts - now) + 0.01)
            st.sec_remaining = max(0, st.sec_remaining - 1)
            st.min_remaining = max(0, st.min_remaining - 1)

    def after(self, headers: dict[str, str]) -> None:
        rem = headers.get("Remaining-Req") or headers.get("remaining-req") or ""
        if not rem:
            return
        # 예: "group=market; min=599; sec=9"
        group = "default"
        sec = None
        minute = None
        parts = [p.strip() for p in rem.split(";")]
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if k == "group":
                group = v
            elif k == "sec":
                try:
                    sec = int(v)
                except ValueError:
                    sec = None
            elif k == "min":
                try:
                    minute = int(v)
                except ValueError:
                    minute = None

        now = time.monotonic()
        with self._lock:
            st = self._states.setdefault(group, _RateState())
            if sec is not None:
                st.sec_remaining = sec
                st.sec_reset_ts = now + 1.05  # 초 경과 버퍼
            if minute is not None:
                st.min_remaining = minute
                st.min_reset_ts = now + 60.05  # 분 경과 버퍼


# ── 내부 유틸 ────────────────────────────────────────────────────────────────

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _fmt_decimal(x: float, max_dp: int = 8) -> str:
    s = f"{x:.{max_dp}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _resolve_tick(price: float, bands: list[tuple[float, float]]) -> float:
    """가격대별 호가단위(bands=[(threshold,tick),...], 오름차순)."""
    tick = 1.0
    for th, tk in bands:
        if price >= th:
            tick = tk
        else:
            break
    return float(tick)


def _round_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return float(price)
    return round(round(float(price) / tick) * tick, 8)


# ── 어댑터 구현 ──────────────────────────────────────────────────────────────

class UpbitBrokerAdapter:
    """
    주문/잔고/시세(Upbit REST).
    - KRW 최소 주문 총액 5000원 검증.
    - 가격단위 라운딩(기본 bands 제공, 사용자 bands로 대체 가능).
    """

    # 기본 tick bands(예시, KRW 호가체계 통상 적용)
    DEFAULT_TICK_BANDS: list[tuple[float, float]] = [
        (0, 0.01), (10, 0.1), (100, 1), (1000, 5), (10000, 10), (100000, 50), (500000, 100)
    ]

    def __init__(
        self,
        *,
        access_key: str,
        secret_key: str,
        tick_bands: Optional[list[tuple[float, float]]] = None,
    ) -> None:
        self.access_key = access_key
        self.secret_key = secret_key
        self.tick_bands = sorted(tick_bands or self.DEFAULT_TICK_BANDS, key=lambda x: x[0])
        self._rl = _RateLimiter()

    # ----- 공개 API -----

    def place_order(self, req: OrderRequest) -> OrderResult:
        market = self._parse_market(req.symbol)
        side = req.side.lower()
        if side not in ("buy", "sell"):
            raise BrokerError("side must be 'buy' or 'sell'", code="BAD_ARG", req={"side": req.side})

        up_side = "bid" if side == "buy" else "ask"

        # 매수 시장가는 금액 기반(ord_type='price') → 안전상 제한
        if req.price is None and side == "buy":
            raise BrokerError("Upbit 시장가 매수는 금액(price) 기반입니다. 제한가를 지정하세요.", code="MARKET_BUY_UNSUPPORTED")

        if req.price is None and side == "sell":
            payload = {"market": market, "side": up_side, "volume": _fmt_decimal(req.qty), "ord_type": "market"}
        else:
            price = float(req.price)  # type: ignore[arg-type]
            tk = _resolve_tick(price, self.tick_bands)
            px_rounded = _round_to_tick(price, tk)
            if market.startswith("KRW-") and (px_rounded * float(req.qty) < 5000.0):
                raise BrokerError("KRW 마켓 최소 주문 총액은 5000원 이상이어야 합니다.", code="MIN_NOTIONAL")
            payload = {
                "market": market,
                "side": up_side,
                "volume": _fmt_decimal(req.qty),
                "price": _fmt_decimal(px_rounded),
                "ord_type": "limit",
            }

        headers = self._auth_headers(payload)
        data = self._request_json("POST", "/v1/orders", headers=headers, data=payload, group="order")
        oid = str(data.get("uuid")) if isinstance(data, dict) else None
        return OrderResult(ok=True, order_id=oid, raw=data, request={"symbol": req.symbol, "side": req.side, "qty": req.qty, "price": req.price})

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        if not order_id:
            raise BrokerError("order_id required", code="BAD_ARG")
        params = {"uuid": order_id}
        headers = self._auth_headers(params)
        return self._request_json("DELETE", "/v1/order", headers=headers, params=params, group="order")  # type: ignore[return-value]

    def fetch_positions(self) -> dict[str, Any]:
        headers = self._auth_headers({})
        data = self._request_json("GET", "/v1/accounts", headers=headers, group="account")
        return {"accounts": data}  # 상위 계층에서 해석

    def fetch_cash(self) -> dict[str, Any]:
        headers = self._auth_headers({})
        data = self._request_json("GET", "/v1/accounts", headers=headers, group="account")
        krw_total = 0.0
        if isinstance(data, list):
            for item in data:
                if item.get("currency") == "KRW":
                    try:
                        bal = float(item.get("balance", 0.0))
                        locked = float(item.get("locked", 0.0))
                    except (TypeError, ValueError):
                        bal = locked = 0.0
                    krw_total = bal + locked
                    break
        return {"KRW": krw_total, "raw": data}

    def fetch_price(self, symbol: str) -> tuple[float, dict[str, Any]]:
        market = self._parse_market(symbol)
        params = {"markets": market}
        data = self._request_json("GET", "/v1/ticker", params=params, group="market")
        if not isinstance(data, list) or not data:
            raise BrokerError("empty ticker response", code="EMPTY", raw={"symbol": symbol})
        trade_price = float(data[0].get("trade_price", 0.0))
        return trade_price, {"ticker": data[0]}

    # ----- 내부 유틸 -----

    @staticmethod
    def _parse_market(symbol: str) -> str:
        """'UPBIT:KRW-BTC' | 'KRW-BTC' → 'KRW-BTC'."""
        s = symbol.strip().upper()
        return s.split(":", 1)[1] if ":" in s else s

    def _auth_headers(self, params_or_body: dict[str, Any]) -> dict[str, str]:
        """사설 엔드포인트 인증 헤더(JWT)."""
        payload: dict[str, Any] = {"access_key": self.access_key, "nonce": str(uuid.uuid4())}
        if params_or_body:
            q = urllib.parse.urlencode(params_or_body).encode("utf-8")
            payload["query_hash"] = hashlib.sha512(q).hexdigest()
            payload["query_hash_alg"] = "SHA512"
        token = self._jwt_sign(payload)
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}

    def _jwt_sign(self, payload: dict[str, Any]) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        h = _b64url(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        p = _b64url(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        signing_input = f"{h}.{p}".encode("ascii")
        sig = hmac.new(self.secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
        return f"{h}.{p}.{_b64url(sig)}"

    def _request_json(
        self,
        method: str,
        path: str,
        headers: Optional[dict[str, str]] = None,
        *,
        params: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
        group: str = "default",
        timeout: int = 10,
    ) -> dict[str, Any] | list[Any]:
        """HTTP JSON 요청 + 레이트리밋 적용 + 에러 승격."""
        self._rl.before(group)

        url = API_HOST + path
        if params:
            url += "?" + urllib.parse.urlencode(params)

        hdrs = {"Accept": "application/json"}
        if headers:
            hdrs.update(headers)

        req = urllib.request.Request(url=url, method=method, headers=hdrs)
        if data is not None:
            req.data = json.dumps(data).encode("utf-8")

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                self._rl.after({k: v for k, v in resp.headers.items()})
                if not raw:
                    raise BrokerError(f"empty response: {getattr(resp, 'status', 'unknown')}", code="EMPTY_HTTP")
                try:
                    obj: dict[str, Any] | list[Any] = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError as je:
                    raise BrokerError(f"json decode error: {url} {je}", code="BAD_JSON") from None
        except HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            code = f"HTTP_{e.code}"
            try:
                obj = json.loads(body)
            except Exception:
                raise BrokerError(f"{code} {url} {body}", code=code) from None
            raise self._normalize_error(obj, code=code, url=url) from None
        except URLError as e:
            raise BrokerError(f"connection error: {url} {e.reason}", code="NETWORK") from None

        if isinstance(obj, dict) and "error" in obj:
            raise self._normalize_error(obj, code="API_ERROR", url=url)
        return obj

    @staticmethod
    def _normalize_error(obj: dict[str, Any], *, code: str, url: str) -> BrokerError:
        err = obj.get("error") or {}
        name = str(err.get("name", "") or obj.get("name", ""))
        msg = str(err.get("message", "") or obj.get("message", "") or "request failed")
        full = f"{name} {msg}".strip()
        return BrokerError(full or code, code=name or code, raw=obj, req={"url": url})

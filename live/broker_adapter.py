# live/broker_adapter.py
"""
KIS(한국투자증권) Open API 실매매 어댑터.

공개 계약
- place_order(OrderRequest) -> OrderResult
- cancel_order(order_id, market=None) -> dict   (해외는 TR 설정 필요)
- fetch_positions() -> dict
- fetch_cash() -> dict
- fetch_price(symbol) -> (price: float, raw: dict)

규약
- UTC ISO-8601("...Z") 가정.
- 해외는 운영 설정(overseas_conf)로 TR/엔드포인트/틱크기 주입.
- 지정가 틱 라운딩은 가능 시 시뮬레이션 규칙 재사용.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import json
import sys
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError

# Python 3.11+ 요구
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ required.")

# 시뮬레이션 라운딩 재사용(가능 시). 모듈/속성 부재만 허용.
try:
    from simulation.execution import round_price as _sim_round_price  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    _sim_round_price = None


# ---------- 공용 인터페이스 ----------

class BrokerError(RuntimeError):
    """브로커 API 오류. code/raw/req를 포함해 상위에서 분기 가능."""
    def __init__(self, message: str, *, code: Optional[str] = None, raw: Optional[dict] = None, req: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.raw = raw or {}
        self.req = req or {}
    def __str__(self) -> str:
        c = f"[{self.code}] " if self.code else ""
        return f"{c}{super().__str__()}"


class AuthError(BrokerError):
    """인증/토큰 오류."""


@dataclass(frozen=True)
class OrderRequest:
    symbol: str                 # 'KRX:005930' | 'NASD:AAPL' 등
    side: str                   # 'buy' | 'sell'
    qty: int
    price: Optional[float] = None   # None: 시장가
    market: Optional[str] = None    # 해외 거래소 코드(예: 'NASD','NYSE')


@dataclass
class OrderResult:
    ok: bool
    order_id: Optional[str]
    raw: dict[str, Any]
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    request: Optional[dict[str, Any]] = None


class BrokerAdapter:
    """실매매 어댑터 공개 계약(간결)."""
    def place_order(self, req: OrderRequest) -> OrderResult: ...
    def cancel_order(self, order_id: str, *, market: Optional[str] = None) -> dict[str, Any]: ...
    def fetch_positions(self) -> dict[str, Any]: ...
    def fetch_cash(self) -> dict[str, Any]: ...
    def fetch_price(self, symbol: str) -> tuple[float, dict[str, Any]]: ...


# ---------- 한국투자증권(KIS) 어댑터 ----------

class KisBrokerAdapter(BrokerAdapter):
    """
    - OAuth2 토큰 자동 갱신.
    - Hashkey 발급 후 주문 헤더에 첨부.
    - use_paper=True → 모의(VTS), False → 실전(PROD).
    """

    PROD_BASE = "https://openapi.koreainvestment.com:9443"
    PAPER_BASE = "https://openapivts.koreainvestment.com:29443"
    TOKEN_PATH = "/oauth2/tokenP"
    HASHKEY_PATH = "/uapi/hashkey"

    # 국내 시세/주문/잔고
    KRX_PRICE_PATH = "/uapi/domestic-stock/v1/quotations/inquire-price"
    KRX_ORDER_CASH_PATH = "/uapi/domestic-stock/v1/trading/order-cash"
    KRX_BALANCE_PATH = "/uapi/domestic-stock/v1/trading/inquire-balance"
    KRX_PSBL_ORDER_PATH = "/uapi/domestic-stock/v1/trading/inquire-psbl-order"

    # 국내 TR_ID(실전/모의 자동 전환)
    TR_PRICE = "FHKST01010100"
    TR_BALANCE = "TTTC8434R"
    TR_PSBL = "TTTC8908R"
    TR_BUY_CASH = ("TTTC0802U", "VTTC0802U")
    TR_SELL_CASH = ("TTTC0801U", "VTTC0801U")

    HTTP_TIMEOUT = 15  # seconds

    def __init__(
        self,
        *,
        app_key: str,
        app_secret: str,
        cano: str,             # 계좌 앞 8자리
        acnt_prdt_cd: str,     # 계좌 뒤 2자리(예: '01')
        use_paper: bool = False,
        overseas_conf: Optional[dict[str, Any]] = None,
    ) -> None:
        self.app_key = app_key
        self.app_secret = app_secret
        self.cano = cano
        self.acnt_prdt_cd = acnt_prdt_cd
        self.use_paper = use_paper
        self.base = self.PAPER_BASE if use_paper else self.PROD_BASE
        self._access_token: Optional[str] = None
        self._exp_ts: float = 0.0
        self.overseas_conf = overseas_conf or {}

    # ----- 공개 API -----

    def place_order(self, req: OrderRequest) -> OrderResult:
        is_domestic, sym, market = self._parse_symbol(req.symbol, req.market)
        if req.side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if req.qty <= 0:
            raise ValueError("qty must be > 0")
        if (req.price is not None) and (req.price < 0):
            raise ValueError("price must be >= 0")
        return (
            self._place_order_krx(req.side, sym, req.qty, req.price)
            if is_domestic
            else self._place_order_overseas(sym, market, req.side, req.qty, req.price)
        )

    def cancel_order(self, order_id: str, *, market: Optional[str] = None) -> dict[str, Any]:
        # 시장/시간대별 TR_ID 상이 → 운영 설정 후 구현
        raise NotImplementedError("취소/정정: 시장별 TR_ID 설정 후 구현하세요.")

    def fetch_positions(self) -> dict[str, Any]:
        self._ensure_token()
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        headers = self._headers(self.TR_BALANCE, need_auth=True)
        return self._request_json("GET", self.KRX_BALANCE_PATH, headers, params=params)

    def fetch_cash(self) -> dict[str, Any]:
        # 규격상 특정 종목 단가 입력 필요 → 주문 직전 대상 종목으로 재호출 권장
        self._ensure_token()
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": "005930",   # placeholder
            "ORD_UNPR": "0",
            "ORD_DVSN": "01",
            "CMA_EVLU_AMT_ICLD_YN": "Y",
            "OVRS_ICLD_YN": "Y",
        }
        headers = self._headers(self.TR_PSBL, need_auth=True)
        return self._request_json("GET", self.KRX_PSBL_ORDER_PATH, headers, params=params)

    def fetch_price(self, symbol: str) -> tuple[float, dict[str, Any]]:
        is_domestic, sym, market = self._parse_symbol(symbol, None)
        if not is_domestic:
            return self._fetch_price_overseas(sym, market)
        self._ensure_token()
        params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": sym}
        headers = self._headers(self.TR_PRICE, need_auth=True)
        data = self._request_json("GET", self.KRX_PRICE_PATH, headers, params=params)
        price = float(data["output"]["stck_prpr"])
        return price, data

    # ----- 오류 매핑 -----
    @staticmethod
    def _normalize_order_error(raw: dict, req: dict) -> BrokerError:
        """KIS 응답(raw)에서 표준 오류로 변환. 잔고/예수금 부족은 code='INSUFFICIENT_CASH'."""
        msg_cd = str(raw.get("msg_cd") or raw.get("rt_cd") or "")
        msg1 = str(raw.get("msg1") or raw.get("msg") or "")
        outmsg = msg1 or msg_cd or "order rejected"
        text = f"{msg_cd} {msg1}".upper()
        insufficient_patterns = ("잔고부족", "예수금부족", "주문가능금액 부족", "부족", "INSUFFICIENT", "NOT ENOUGH CASH", "INSUFFICIENT FUNDS")
        code = "INSUFFICIENT_CASH" if any(pat in text or pat in outmsg for pat in insufficient_patterns) else "REJECTED"
        return BrokerError(outmsg.strip(), code=code, raw=raw, req=req)

    # ----- 국내 -----

    def _place_order_krx(self, side: str, pdno: str, qty: int, price: Optional[float]) -> OrderResult:
        self._ensure_token()
        is_market = price is None
        ord_dvsn = "01" if is_market else "00"
        payload = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": pdno,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": "0" if is_market else str(price),
        }
        tr = self._pick_tr(self.TR_BUY_CASH if side == "buy" else self.TR_SELL_CASH)
        headers = self._headers(tr, need_auth=True, with_hashkey=payload)
        raw = self._request_json("POST", self.KRX_ORDER_CASH_PATH, headers, data=payload)
        # 성공/실패 판정: KIS는 HTTP 200에서도 rt_cd/msg_cd로 거절을 표기
        req_echo = {"side": side, "pdno": pdno, "qty": int(qty), "price": price}
        rt = str(raw.get("rt_cd", "0"))
        if rt not in ("0", "OPERATION", "SUCCESS"):
            raise self._normalize_order_error(raw, req_echo)
        oid = raw.get("output", {}).get("ORD_NO") or raw.get("output", {}).get("ODNO")
        return OrderResult(ok=True, order_id=oid, raw=raw, request=req_echo)

    # ----- 해외 -----

    def _place_order_overseas(
        self, pdno: str, market: Optional[str], side: str, qty: int, price: Optional[float]
    ) -> OrderResult:
        if market is None:
            raise BrokerError("해외 주문은 market 코드가 필요합니다(예: 'NASD','NYSE').")
        oc = self.overseas_conf
        order_path = oc.get("order_path")
        tr_map = oc.get("tr", {}).get("buy" if side == "buy" else "sell", {})
        tr_id = tr_map.get(market)
        if not (order_path and tr_id):
            raise BrokerError(f"해외 주문 설정 누락: order_path/tr_id for market={market}")

        # 지정가일 때만 틱 라운딩
        is_market = price is None
        use_price = price
        if (not is_market) and (_sim_round_price is not None):
            step = self._resolve_overseas_price_step(pdno=pdno, market=market)
            if step and step > 0:
                rp = _sim_round_price(float(price), float(step), mode="nearest")
                if rp is not None:
                    use_price = float(rp)

        self._ensure_token()
        payload = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "OVRS_EXCG_CD": market,
            "PDNO": pdno,
            "ORD_QTY": str(int(qty)),
            "OVRS_ORD_UNPR": "0" if is_market else str(use_price),
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "01" if is_market else "00",
        }
        headers = self._headers(tr_id, need_auth=True, with_hashkey=payload)
        raw = self._request_json("POST", order_path, headers, data=payload)
        req_echo = {"pdno": pdno, "market": market, "side": side, "qty": int(qty), "price": use_price}
        rt = str(raw.get("rt_cd", "0"))
        if rt not in ("0", "OPERATION", "SUCCESS"):
            raise self._normalize_order_error(raw, req_echo)
        oid = raw.get("output", {}).get("ODNO") or raw.get("output", {}).get("ORD_NO")
        return OrderResult(ok=True, order_id=oid, raw=raw, request=req_echo)

    def _fetch_price_overseas(self, pdno: str, market: Optional[str]) -> tuple[float, dict[str, Any]]:
        """해외 현재가: /uapi/overseas-price/v1/quotations/price (EXCD/SYMB 사용)."""
        if market is None:
            raise BrokerError("해외 시세는 market 코드가 필요합니다.")

        # 기본 경로/트랜잭션 ID(설정에서 덮어쓰기 가능)
        oc = self.overseas_conf or {}
        price_path = oc.get("price_path") or "/uapi/overseas-price/v1/quotations/price"
        default_tr = "HHDFS00000300"  # 해외 현재가(REST)
        tr_id = oc.get("tr", {}).get("price") or default_tr

        # 거래소 코드 4→3글자 매핑
        exch_map = {"NASD": "NAS", "NYSE": "NYS", "NAS": "NAS", "NYS": "NYS", "AMEX": "AMEX"}
        excd = exch_map.get(market, market)

        self._ensure_token()
        params = {"AUTH": "", "EXCD": excd, "SYMB": pdno}
        headers = self._headers(tr_id, need_auth=True)
        data = self._request_json("GET", price_path, headers, params=params)

        # 응답 포맷 호환: 'last' 우선, 없으면 'ovrs_prpr'
        output = data.get("output", {}) if isinstance(data, dict) else {}
        price = float(output.get("last", output.get("ovrs_prpr", 0.0)))
        return price, data

    # ----- 내부 유틸 -----

    def _pick_tr(self, pair: tuple[str, str]) -> str:
        return pair[1] if self.use_paper else pair[0]

    def _parse_symbol(self, symbol: str, market_hint: Optional[str]) -> tuple[bool, str, Optional[str]]:
        """'KRX:005930' → (True,'005930',None), 'NASD:AAPL' → (False,'AAPL','NASD')."""
        if ":" not in symbol:
            return True, symbol, None
        prefix, code = symbol.split(":", 1)
        if prefix.upper() in ("KRX", "KOSPI", "KOSDAQ"):
            return True, code, None
        return False, code, market_hint or prefix.upper()

    def _resolve_overseas_price_step(self, *, pdno: str, market: str) -> Optional[float]:
        """틱 크기 조회 우선순위: '{market}:{pdno}' > '{pdno}' > '{market}'."""
        step_map = self.overseas_conf.get("price_step", {})
        if not isinstance(step_map, dict):
            return None
        key1 = f"{market}:{pdno}"
        step = step_map.get(key1) or step_map.get(pdno) or step_map.get(market)
        try:
            return float(step) if step is not None else None
        except (TypeError, ValueError):
            return None

    def _headers(self, tr_id: str, *, need_auth: bool, with_hashkey: Optional[dict] = None) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        if need_auth:
            if not self._access_token:
                raise AuthError("access_token 없음")
            h["authorization"] = f"Bearer {self._access_token}"
        if with_hashkey is not None:
            h["hashkey"] = self._issue_hashkey(with_hashkey)
        return h

    def _ensure_token(self) -> None:
        now = time.time()
        if self._access_token and now < (self._exp_ts - 60):  # 만료 60초 전 재발급
            return
        url = self.base + self.TOKEN_PATH
        body = {"grant_type": "client_credentials", "appkey": self.app_key, "appsecret": self.app_secret}
        data = self._http_json("POST", url, headers={"Content-Type": "application/json"}, data=body)
        token = data.get("access_token")
        expires_in = int(data.get("expires_in", 24 * 3600))
        if not token:
            raise AuthError(f"토큰 발급 실패: {data}")
        self._access_token = token
        self._exp_ts = time.time() + expires_in

    def _issue_hashkey(self, payload: dict) -> str:
        url = self.base + self.HASHKEY_PATH
        headers = {"Content-Type": "application/json", "appkey": self.app_key, "appsecret": self.app_secret}
        data = self._http_json("POST", url, headers=headers, data=payload)
        hk = data.get("HASH") or data.get("hash")
        if not hk:
            raise BrokerError(f"hashkey 발급 실패: {data}")
        return hk

    def _request_json(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        *,
        params: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        return self._http_json(method, url, headers=headers, data=data)

    def _http_json(self, method: str, url: str, headers: dict[str, str], data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        req = urllib.request.Request(url=url, method=method, headers=headers)
        if data is not None:
            req.data = json.dumps(data).encode("utf-8")
        try:
            with urllib.request.urlopen(req, timeout=self.HTTP_TIMEOUT) as resp:
                raw = resp.read()
                if not raw:
                    raise BrokerError(f"빈 응답: {getattr(resp, 'status', 'unknown')}")
                try:
                    return json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError as je:
                    raise BrokerError(f"JSON 디코드 오류: {url} {je}") from None
        except HTTPError as e:
            detail = e.read().decode("utf-8") if e.fp else ""
            raise BrokerError(f"HTTP {e.code} {url} {detail}") from None
        except URLError as e:
            raise BrokerError(f"연결 오류: {url} {e.reason}") from None

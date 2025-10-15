# data/universe.py
"""
자산군별 유니버스(후보군) 생성 → 공통 스키마 DataFrame 반환.

공개 API
- build_universe(asset_class, cfg) -> pd.DataFrame
  * asset_class: "upbit" | "us" | "krx"
  * 반환 컬럼(가능 시):
    [symbol, asset_class, currency, tick_size, lot_step, trading_status,
     liquidity_24h, ref_price, meta]

규약
- tick_size/lot_step: 주문 라운딩·수량 단위. 가격대별 단위는 tick_table로 계산.
- liquidity_24h: 24시간 누적 거래대금(가격통화 기준) — 가능 시 채움.
- trading_status: "active" | "warning" | "halted" | "unknown".

예외 처리
- 외부 호출/파싱 오류는 하위 유틸(collect.py)에서 RuntimeError로 승격. 여기서는 그대로 전파.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json
import math
import pandas as pd

# 로컬 수집 유틸
from .collect import (
    list_upbit_markets,
    fetch_upbit_tickers,
    list_us_equities_from_nasdaq_dir,
)

__all__ = ["build_universe", "resolve_tick", "load_tick_table"]


# ── Tick/Lot 유틸 ─────────────────────────────────────────────────────────────

def load_tick_table(cfg: dict[str, Any], key: str) -> list[tuple[float, float]]:
    """
    cfg에서 가격대별 tick 테이블을 로드.
    - 입력:
      1) 리스트 [[threshold, tick], ...]
      2) 파일 경로(JSON: {"bands": [[threshold, tick], ...]})
    - 반환: 오름차순 정렬된 [(threshold, tick), ...]
    """
    src = (cfg or {}).get(key)
    bands: list[tuple[float, float]] = []
    if not src:
        return bands

    if isinstance(src, list):
        bands = [(float(a), float(b)) for a, b in src]  # type: ignore[misc]
    else:
        p = Path(str(src))
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        raw = data.get("bands") or data.get("tick_bands") or []
        bands = [(float(a), float(b)) for a, b in raw]

    bands.sort(key=lambda x: x[0])
    return bands


def resolve_tick(price: float | None, bands: list[tuple[float, float]], default_tick: float) -> float:
    """가격대별 tick 테이블에 따라 tick_size 계산. 가격/테이블 미제공 시 default_tick."""
    if price is None or not math.isfinite(price) or price <= 0 or not bands:
        return float(default_tick)
    tick = default_tick
    for th, tk in bands:
        if price >= th:
            tick = tk
        else:
            break
    return float(tick)


# ── 스키마 ────────────────────────────────────────────────────────────────────

@dataclass
class Row:
    symbol: str
    asset_class: str
    currency: str | None = None
    tick_size: float | None = None
    lot_step: float | None = None
    trading_status: str = "unknown"
    liquidity_24h: float | None = None
    ref_price: float | None = None
    meta: dict[str, Any] | None = None  # JSON 직렬화 가능


# ── 자산군 빌더 ────────────────────────────────────────────────────────────────

def _build_upbit(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Upbit KRW 마켓 유니버스.
    - 가격단위: cfg["rounding"]["upbit_tick_table"] 사용(없으면 1.0 KRW).
    - 최소주문: 5,000 KRW(meta로 제공).
    - 유동성: acc_trade_price_24h.
    - 상태: market_event/market_warning로 warning 판정.
    """
    quote = (cfg.get("universe", {}).get("upbit", {}) or {}).get("quote", "KRW")
    markets = list_upbit_markets(quote=quote)
    if markets.empty:
        return pd.DataFrame(columns=Row.__dataclass_fields__.keys())

    mlist = markets["market"].astype(str).tolist()
    tk_df = fetch_upbit_tickers(mlist) if mlist else pd.DataFrame()

    rounding_cfg = cfg.get("rounding", {}) or {}
    upbit_bands = load_tick_table(rounding_cfg, "upbit_tick_table")  # 예: [[0,1],[100,5],... KRW 단위]
    upbit_min_notional = float(rounding_cfg.get("upbit_min_notional", 5000.0))
    lot_step = float((cfg.get("lot_step", {}) or {}).get("crypto", 1e-8))

    by_market = tk_df.set_index("market") if not tk_df.empty else pd.DataFrame()

    rows: list[Row] = []
    for _, r in markets.iterrows():
        mkt = str(r["market"])
        t = by_market.loc[mkt] if not by_market.empty and mkt in by_market.index else None

        trade_price = float(t["trade_price"]) if t is not None and "trade_price" in t else None
        tick_size = resolve_tick(trade_price, upbit_bands, default_tick=1.0)

        warning = False
        ev = r.get("market_event")
        if isinstance(ev, dict):
            warning = bool(ev.get("warning") or ev.get("caution"))
        warn_field = str(r.get("market_warning", "")).upper()
        warning = warning or (warn_field == "CAUTION")

        rows.append(
            Row(
                symbol=f"UPBIT:{mkt}",
                asset_class="crypto",
                currency=mkt.split("-", 1)[0] if "-" in mkt else quote,
                tick_size=tick_size,
                lot_step=lot_step,
                trading_status="warning" if warning else "active",
                liquidity_24h=float(t["acc_trade_price_24h"]) if t is not None and "acc_trade_price_24h" in t else None,
                ref_price=trade_price,
                meta={"upbit_min_notional": upbit_min_notional, "tick_table_bands": upbit_bands},
            )
        )

    df = pd.DataFrame([asdict(x) for x in rows])
    return df.sort_values("symbol").reset_index(drop=True)


def _build_us(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    미국 상장주식 유니버스.
    - 소스: NASDAQ Symbol Directory(nasdaqlisted/otherlisted) → Test Issue 제외.
    - tick_size: 0.01(기본), lot_step: 1, currency: USD.
    """
    master = list_us_equities_from_nasdaq_dir()
    if master.empty:
        return pd.DataFrame(columns=Row.__dataclass_fields__.keys())

    default_tick = float((cfg.get("rounding", {}) or {}).get("us_default_tick", 0.01))
    lot_step = float((cfg.get("lot_step", {}) or {}).get("equity_us", 1))

    rows: list[Row] = []
    for _, r in master.iterrows():
        exch = str(r.get("exchange", "US")).upper()
        sym = str(r["symbol"]).upper()
        rows.append(
            Row(
                symbol=f"{exch}:{sym}",
                asset_class="equity_us",
                currency="USD",
                tick_size=default_tick,
                lot_step=lot_step,
                trading_status="active",
                liquidity_24h=None,
                ref_price=None,
                meta={
                    "security_name": r.get("security_name"),
                    "is_etf": bool(r.get("is_etf", False)),
                    "source_file": r.get("source_file"),
                    "file_creation_time": r.get("file_creation_time"),
                },
            )
        )

    df = pd.DataFrame([asdict(x) for x in rows])
    excl_etf = bool((cfg.get("universe", {}).get("us", {}) or {}).get("exclude_etf", False))
    if excl_etf and "meta" in df.columns:
        df = df[df["meta"].map(lambda m: not (m or {}).get("is_etf", False))]
    return df.sort_values("symbol").reset_index(drop=True)


def _build_krx(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    KRX 유니버스(심볼 소스는 cfg로 주입).
    - tick_size: cfg.rounding.krx_tick_table 가격대별 계산(가격대별 호가단위).
    - lot_step: 1, currency: KRW.
    - ref_price: cfg['universe']['krx']['reference_price_map'][symbol] 사용 시 tick 계산.
    """
    ucfg = (cfg.get("universe", {}).get("krx", {}) or {})
    symbols: list[str] = list(ucfg.get("symbols", []))  # 예: ["KRX:005930", "KRX:000660"]
    if not symbols:
        return pd.DataFrame(columns=Row.__dataclass_fields__.keys())

    bands = load_tick_table(cfg.get("rounding", {}) or {}, "krx_tick_table")
    lot_step = float((cfg.get("lot_step", {}) or {}).get("equity_kr", 1))
    ref_price_map: dict[str, float] = (ucfg.get("reference_price_map") or {})

    rows: list[Row] = []
    for s in symbols:
        price = ref_price_map.get(s)
        tick = resolve_tick(price, bands, default_tick=1.0)
        rows.append(
            Row(
                symbol=s,
                asset_class="equity_kr",
                currency="KRW",
                tick_size=tick,
                lot_step=lot_step,
                trading_status="active",
                liquidity_24h=None,
                ref_price=price,
                meta={"tick_table_bands": bands},
            )
        )
    return pd.DataFrame([asdict(x) for x in rows]).sort_values("symbol").reset_index(drop=True)


# ── 파사드 ────────────────────────────────────────────────────────────────────

def build_universe(asset_class: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """
    자산군별 유니버스를 공통 스키마로 반환.
    - asset_class: "upbit" | "us" | "krx"
    - cfg 예시 키:
        universe.upbit.quote
        rounding.upbit_tick_table / rounding.upbit_min_notional
        rounding.krx_tick_table
        lot_step.crypto / lot_step.equity_us / lot_step.equity_kr
        universe.us.exclude_etf
        universe.krx.symbols, universe.krx.reference_price_map
    """
    ac = asset_class.lower()
    if ac == "upbit":
        return _build_upbit(cfg or {})
    if ac == "us":
        return _build_us(cfg or {})
    if ac == "krx":
        return _build_krx(cfg or {})
    raise ValueError(f"unsupported asset_class: {asset_class}")

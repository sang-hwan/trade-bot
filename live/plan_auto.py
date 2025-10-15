# live/plan_auto.py
"""
AUTO 계획 산출 모듈 (Python 3.11+)

공개 API
- build_plan_auto(cfg, *, source, interval) -> (stops_list, signals_map, rebal_spec)

타이밍 규약
- 룩어헤드 금지: D-1 종가(UTC)까지만 사용해 당일(D) 시가 기준 계획을 산출.

동작 개요
- data.universe.build_universe()로 후보 풀 생성 → 거래상태/유동성 1차 필터 →
  Stop/Signal 계산 및 수량 산정 → 감사 플랜(plan/plan.json) 기록.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import pandas as pd


# ── 내부 유틸 ────────────────────────────────────────────────────────────────

def _round_price_nearest(price: float, step: float) -> float:
    """틱(step) 기준 최근접 반올림(동률=올림). step<=0이면 원값."""
    step = float(step or 0.0)
    if step <= 0.0 or price <= 0.0:
        return float(price)
    q = Decimal(str(step))
    ticks = (Decimal(str(price)) / q).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return float((ticks * q).quantize(q, rounding=ROUND_HALF_UP))


def _build_candidate_symbols(cfg: dict[str, Any]) -> list[str]:
    """
    유니버스 기반 후보 풀 생성 후 1차 필터링.
    필터: trading_status(거래정지/경보 제외), min_liquidity_24h 이상.
    """
    from data.universe import build_universe  # 지연 import

    uv_cfg = dict(cfg.get("universe", {}))
    asset_classes = uv_cfg.get("asset_class", ["equity_us"])
    if isinstance(asset_classes, (str, bytes)):
        asset_classes = [asset_classes]

    min_liq = float(uv_cfg.get("min_liquidity_24h", 0.0))
    allow_status = set(uv_cfg.get("allow_trading_status", {"TRADING", "ACTIVE", "LISTED"}))
    deny_status = set(uv_cfg.get("deny_trading_status", {"HALTED", "SUSPENDED", "WARN", "WARNING", "CAUTION"}))
    max_candidates = int(uv_cfg.get("max_candidates", 2000))

    frames: list[pd.DataFrame] = []
    for ac in asset_classes:
        try:
            df = build_universe(ac, uv_cfg)
        except Exception as e:
            print(f"[WARN] universe build failed for asset_class={ac}: {e}", file=sys.stderr)
            continue
        if isinstance(df, pd.DataFrame) and not df.empty:
            frames.append(df)

    if not frames:
        return []

    u = pd.concat(frames, ignore_index=True)

    # 표준 컬럼 가정: symbol, trading_status, liquidity_24h
    if "trading_status" in u.columns:
        ts = u["trading_status"].astype(str).str.upper()
        mask = (~ts.isin(deny_status)) & (ts.isin(allow_status) if allow_status else True)
        u = u.loc[mask]

    if "liquidity_24h" in u.columns and min_liq > 0.0:
        with pd.option_context("mode.use_inf_as_na", True):
            u = u.loc[(pd.to_numeric(u["liquidity_24h"], errors="coerce").fillna(0.0) >= min_liq)]

    if "liquidity_24h" in u.columns:
        u = u.sort_values("liquidity_24h", ascending=False)

    if max_candidates > 0:
        u = u.head(max_candidates)

    syms = [str(s) for s in u.get("symbol", []) if isinstance(s, (str, bytes))]
    return syms


# ── 공개 API ────────────────────────────────────────────────────────────────

def build_plan_auto(
    cfg: dict[str, Any], *, source: str, interval: str
) -> tuple[list[str], dict[str, int], dict[str, dict[str, float]]]:
    """
    D-1 종가(UTC)까지의 데이터로 당일(D) 시가 집행 계획 산출.

    반환:
      - stops_list : 오늘 시가 청산 대상 심볼 리스트
      - signals_map: 오늘 시가 진입 목표 수량(주)
      - rebal_spec : 현금 기준 리밸 금액 맵 {"buy_notional":{}, "sell_notional":{}}
    """
    # 전략/데이터 계층 지연 import
    from data.collect import collect
    from data.quality_gate import validate as qv
    from data.adjust import apply as adj
    from strategy.signals import sma_cross_long_only
    from strategy.stops import donchian_stop_long
    from strategy.sizing_spec import build_fixed_fractional_spec
    from simulation.sizing_on_open import size_from_spec
    import datetime as dt

    LOOKBACK_YEARS = 10

    base_ccy = cfg.get("base_currency", "USD")
    cal_id = cfg.get("calendar_id", "XNAS")
    eng = cfg.get("engine", {})
    N = int(eng.get("N", 20))
    f = float(eng.get("f", 0.02))
    lot_step = float(eng.get("lot_step", 1))
    epsilon = float(eng.get("epsilon", 0.0))
    sma_short = int(eng.get("sma_short", 10))
    sma_long = int(eng.get("sma_long", 50))
    initial_cash = float(cfg.get("initial_cash", 10_000.0))

    # [1] 후보 풀 구성(유니버스) + 1차 필터
    target_symbols = _build_candidate_symbols(cfg)

    price_step = float(cfg.get("price_step", 0.0))
    price_cushion_pct = 0.001
    fee_cushion_pct = 0.0005

    today_utc = dt.datetime.utcnow().date()
    end = today_utc - dt.timedelta(days=1)
    start = end - dt.timedelta(days=365 * LOOKBACK_YEARS)

    stops_list: list[str] = []
    signals_map: dict[str, int] = {}
    rebal_spec: dict[str, dict[str, float]] = {"buy_notional": {}, "sell_notional": {}}
    audit_signals: list[dict[str, Any]] = []

    for symbol in target_symbols:
        y_ticker = symbol.split(":", 1)[-1]

        try:
            cr = collect(
                source, y_ticker, start=start.isoformat(), end=end.isoformat(),
                interval=interval, base_currency=base_ccy, calendar_id=cal_id,
            )
            qv(cr.dataframe, require_volume=False, min_rows=sma_long + 5)
            df = adj(cr.dataframe)
        except Exception as e:
            print(f"[WARN] {symbol}: 데이터 처리 중 오류, 건너뜀 ({e})", file=sys.stderr)
            continue

        if df.empty:
            continue

        last_ts = df.index.max()

        # Stop: D-1 종가 판단 → D 시가 청산
        st = donchian_stop_long(df, N)
        if (last_ts in st.index) and bool(st.loc[last_ts, "stop_hit"]):
            stops_list.append(symbol)

        # Signal: D-1 종가 판단 → D 시가 진입
        sig = sma_cross_long_only(df, short=sma_short, long=sma_long, epsilon=epsilon)
        will_enter = (
            (last_ts in sig.index)
            and (sig.loc[last_ts] == sig.loc[last_ts])  # NaN 방지
            and (int(sig.loc[last_ts]) == 1)
        )

        # Size: D 시가 미상 → D-1 종가 프록시
        spec = build_fixed_fractional_spec(df, N=N, f=f, lot_step=lot_step)
        stop_level = float(spec.loc[last_ts, "stop_level"])
        price_col = "close_adj" if "close_adj" in df.columns else "close"
        last_close = float(df.loc[last_ts, price_col])

        sz = size_from_spec(
            entry_price=last_close, equity=initial_cash, f=f,
            stop_level=stop_level, lot_step=lot_step, V=None, PV=None,
        )
        q_exec = int(sz.get("Q_exec", 0))
        signals_map[symbol] = q_exec if will_enter else 0

        # 감사용 데이터(쿠션/수수료 적용 추정가)
        price_est = _round_price_nearest(last_close * (1.0 + price_cushion_pct), price_step)
        cash_need = price_est * float(q_exec) * (1.0 + fee_cushion_pct)
        audit_signals.append({
            "symbol": symbol, "requested_qty": q_exec, "price_est": price_est,
            "cash_need": cash_need, "will_enter": bool(will_enter),
        })

    # 감사 플랜 저장(비치명적)
    try:
        plan_dir = "plan"
        os.makedirs(plan_dir, exist_ok=True)
        plan_doc = {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "base_currency": base_ccy,
            "summary": {
                "stops_count": len(stops_list),
                "signals_count": len([k for k, v in signals_map.items() if v > 0]),
                "rebal_has_targets": bool(rebal_spec.get("buy_notional") or rebal_spec.get("sell_notional")),
            },
            "stops": list(stops_list),
            "signals": audit_signals,
            "rebal_summary": rebal_spec,
        }
        with open(os.path.join(plan_dir, "plan.json"), "w", encoding="utf-8") as f:
            json.dump(plan_doc, f, ensure_ascii=False, indent=2)
    except (OSError, TypeError, ValueError) as e:
        print(f"[WARN] 감사 플랜 저장 실패: {e}", file=sys.stderr)

    return stops_list, signals_map, rebal_spec

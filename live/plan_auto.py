# live/plan_auto.py
"""
AUTO 계획 산출 모듈 (Python 3.11+)

공개 API
- build_plan_auto(cfg, *, source, interval) -> (stops_list, signals_map, rebal_spec)

타이밍 규약
- 룩어헤드 금지: D-1 종가(UTC)까지만 사용해 당일(D) 시가 기준 계획을 산출.
"""

from __future__ import annotations

from typing import Any
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
import os
import json


def _round_price_nearest(price: float, step: float) -> float:
    """틱(step) 기준 최근접 반올림(동률=올림). step<=0이면 원값."""
    step = float(step or 0.0)
    if step <= 0.0 or price <= 0.0:
        return float(price)
    q = Decimal(str(step))
    ticks = (Decimal(str(price)) / q).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return float((ticks * q).quantize(q, rounding=ROUND_HALF_UP))


def build_plan_auto(
    cfg: dict[str, Any], *, source: str, interval: str
) -> tuple[list[str], dict[str, int], dict[str, dict[str, float]]]:
    """
    D-1 종가(UTC)까지의 데이터로 당일(D) 시가 집행 계획을 산출.
    반환:
      - stops_list : 오늘 시가 청산 대상 심볼 리스트
      - signals_map: 오늘 시가 진입 목표 수량(주)
      - rebal_spec : 현금 기준 리밸 금액 맵(초기 버전은 빈 구조)
    """
    # 전략/데이터 계층은 AUTO 모드에서만 사용하므로 지연 import
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
    assets = cfg.get("assets", [])

    # 주석 필드 계산 파라미터
    price_step = float(cfg.get("price_step", 0.0))
    price_cushion_pct = 0.001
    fee_cushion_pct = 0.0005

    today_utc = dt.datetime.utcnow().date()
    end = today_utc - dt.timedelta(days=1)  # D-1
    start = end - dt.timedelta(days=365 * LOOKBACK_YEARS)

    stops_list: list[str] = []
    signals_map: dict[str, int] = {}
    rebal_spec: dict[str, dict[str, float]] = {"buy_notional": {}, "sell_notional": {}}

    # 감사용 주석(매수 후보)
    audit_signals: list[dict[str, Any]] = []

    for a in assets:
        symbol = a["symbol"]                  # 예: "NASD:AAPL"
        y_ticker = symbol.split(":")[-1]      # Yahoo 심볼

        # 1) 수집 → 검증 → 조정
        cr = collect(
            source,
            y_ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            interval=interval,
            base_currency=base_ccy,
            calendar_id=cal_id,
        )
        qv(cr.dataframe, require_volume=False, min_rows=50)
        df = adj(cr.dataframe)
        if df.empty:
            signals_map[symbol] = 0
            audit_signals.append({
                "symbol": symbol,
                "requested_qty": 0,
                "price_est": 0.0,
                "cash_need": 0.0,
                "will_enter": False,
            })
            continue

        last_ts = df.index.max()

        # 2) 스탑 판정 (D-1 종가 → D 오픈 청산)
        st = donchian_stop_long(df, N)
        if (last_ts in st.index) and bool(st.loc[last_ts, "stop_hit"]):
            stops_list.append(symbol)

        # 3) 신호 판정 (D-1 종가 → D 오픈 진입)
        sig = sma_cross_long_only(df, short=sma_short, long=sma_long, epsilon=epsilon)
        will_enter = (
            (last_ts in sig.index)
            and (sig.loc[last_ts] == sig.loc[last_ts])  # NaN 방지
            and (int(sig.loc[last_ts]) == 1)
        )

        # 4) 수량 산정 (Fixed Fractional) — D 오픈가 미상 → D-1 종가 프록시
        spec = build_fixed_fractional_spec(df, N=N, f=f, lot_step=lot_step)
        stop_level = float(spec.loc[last_ts, "stop_level"])
        last_close = float(
            df.loc[last_ts, "close_adj"] if "close_adj" in df.columns else df.loc[last_ts, "close"]
        )
        entry_proxy = last_close
        sz = size_from_spec(
            entry_price=entry_proxy,
            equity=initial_cash,
            f=f,
            stop_level=stop_level,
            lot_step=lot_step,
            V=None,
            PV=None,
        )
        q_exec = int(sz.get("Q_exec", 0))
        signals_map[symbol] = q_exec if will_enter else 0

        # --- 매수 후보 주석 필드 ---
        price_est = _round_price_nearest(last_close * (1.0 + price_cushion_pct), price_step)
        cash_need = price_est * float(q_exec) * (1.0 + fee_cushion_pct)
        audit_signals.append({
            "symbol": symbol,
            "requested_qty": q_exec,
            "price_est": price_est,
            "cash_need": cash_need,
            "will_enter": bool(will_enter),
        })

    # 매도 원천 고정: 계획 단계에서 임의 현금 마련 매도 생성 금지
    assert set(rebal_spec.keys()) == {"buy_notional", "sell_notional"}

    # 감사 플랜 저장: plan/plan.json
    try:
        plan_dir = "plan"
        os.makedirs(plan_dir, exist_ok=True)
        plan_doc = {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "base_currency": base_ccy,
            "summary": {
                "stops_count": len(stops_list),
                "signals_count": len(signals_map),
                "rebal_has_targets": bool(rebal_spec.get("buy_notional") or rebal_spec.get("sell_notional")),
            },
            "stops": list(stops_list),
            "signals": audit_signals,
            "rebal_summary": {
                "buy_notional": rebal_spec.get("buy_notional", {}),
                "sell_notional": rebal_spec.get("sell_notional", {}),
            },
        }
        with open(os.path.join(plan_dir, "plan.json"), "w", encoding="utf-8") as f:
            json.dump(plan_doc, f, ensure_ascii=False, indent=2)
    except (OSError, TypeError, ValueError):
        # 계획 저장 실패는 비치명
        pass

    return stops_list, signals_map, rebal_spec

# live/plan_auto.py
"""
AUTO 계획 산출 모듈 (Python 3.11+)

공개 API
- build_plan_auto(cfg, *, source, interval) -> (stops_list, signals_map, rebal_spec)

타이밍 규약
- 룩어헤드 금지: D-1 종가(UTC)까지만 사용해 당일(D) 시가 기준 계획을 산출.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple


def build_plan_auto(
    cfg: Dict[str, Any], *, source: str, interval: str
) -> Tuple[List[str], Dict[str, int], Dict[str, Dict[str, float]]]:
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

    LOOKBACK_YEARS = 10  # 과거 참조 길이(연)

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

    today_utc = dt.datetime.utcnow().date()
    end = today_utc - dt.timedelta(days=1)  # D-1
    start = end - dt.timedelta(days=365 * LOOKBACK_YEARS)

    stops_list: List[str] = []
    signals_map: Dict[str, int] = {}
    rebal_spec: Dict[str, Dict[str, float]] = {"buy_notional": {}, "sell_notional": {}}

    for a in assets:
        symbol = a["symbol"]                  # 예: "NASD:TNDM"
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
        entry_proxy = float(
            df.loc[last_ts, "close_adj"] if "close_adj" in df.columns else df.loc[last_ts, "close"]
        )
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

    return stops_list, signals_map, rebal_spec

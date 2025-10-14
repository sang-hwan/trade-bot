# backtest.py
from __future__ import annotations

# 표준 라이브러리
import argparse
import datetime as _dt
import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

# 서드파티 라이브러리
import pandas as pd

# 프로젝트 모듈
from data.collect import collect
from data.adjust import apply as adj
from data.quality_gate import validate as qv, validate_meta as qv_meta
from data.snapshot import write as snap
from strategy.signals import sma_cross_long_only
from strategy.stops import donchian_stop_long
from strategy.sizing_spec import build_fixed_fractional_spec
from strategy.rebalancing_spec import build_rebalancing_spec_ts
import simulation.engine as eng


def _today_ymd() -> str:
    """현재 UTC 날짜를 YYYY-MM-DD 형식으로 반환한다."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


def _safe_name(symbol: str) -> str:
    """파일 경로에 사용 가능하도록 심볼 이름을 안전하게 변환한다."""
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in symbol)


def _normalize_for_upbit(symbol: str, interval: str) -> tuple[str, str, dict[str, str]]:
    """Upbit API의 입력 형식에 맞게 심볼과 인터벌을 정규화한다."""
    warns: dict[str, str] = {}
    s = symbol.upper().replace(" ", "")
    if "/" in s:
        base, quote = s.split("/", 1)
        s2 = f"{quote}-{base}"
        warns["symbol"] = f"Normalized Upbit symbol '{symbol}' -> '{s2}'"
        s = s2
    elif "-" in s and s.split("-")[0] not in ("KRW", "USDT", "BTC"):
        a, b = s.split("-", 1)
        s2 = f"{b}-{a}"
        warns["symbol"] = f"Normalized Upbit symbol '{symbol}' -> '{s2}'"
        s = s2

    m = {"1D": "day", "1W": "week", "1M": "month"}
    i = m.get(interval.upper(), interval)
    if i != interval:
        warns["interval"] = f"Normalized Upbit interval '{interval}' -> '{i}'"
    return s, i, warns


def _compose_out_dir(root: Path, runs_dir: str, source: str, symbol: str, interval: str, out_dir: str | None) -> Path:
    """백테스트 결과물을 저장할 출력 디렉터리 경로를 생성한다."""
    if out_dir:
        p = Path(out_dir)
        return p if p.is_absolute() else (root / out_dir)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return root / runs_dir / f"{stamp}_{source}_{_safe_name(symbol)}_{interval}"


def run_once(
    *,
    source: str,
    symbol: str,
    start: str,
    end: str,
    interval: str,
    N: int,
    f: float,
    lot_step: float,
    commission_rate: float,
    slip: float,
    epsilon: float,
    initial_equity: float,
    V: float | None,
    PV: float | None,
    snapshot: bool,
    out_dir: Path,
    base_currency: str,
    calendar_id: str | None,
    fx_source: str | None,
    fx_source_ts: str | None,
    instrument_registry_hash: str | None,
    price_step: float,
) -> dict[str, Any]:
    """백테스트 파이프라인을 한 번 실행하는 오케스트레이션 함수."""
    warns: dict[str, str] = {}
    src = source.lower()
    sym, iv = symbol, interval
    if src == "upbit":
        sym, iv, warns = _normalize_for_upbit(symbol, interval)
        if not calendar_id:
            calendar_id = "24x7"

    out_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 수집
    collected = collect(src, sym, start=start, end=end, interval=iv, base_currency=base_currency, calendar_id=calendar_id)
    df = collected.dataframe
    local_ccy = (collected.meta or {}).get("price_currency")

    # 데이터 품질 검증
    qv(df)

    # 가격 조정
    df = adj(df)

    # 메타데이터 검증
    meta_for_validation = {"base_currency": base_currency, "price_currency": local_ccy, "calendar_id": calendar_id, "lot_step": lot_step, "price_step": price_step, "fx_source": fx_source, "fx_source_ts": fx_source_ts}
    qv_meta(meta_for_validation, require_fx=bool(local_ccy and base_currency and local_ccy != base_currency), require_calendar=True, require_steps=price_step > 0.0)

    # 데이터 스냅샷 저장 (선택 사항)
    snapshot_meta: dict[str, Any] | None = None
    if snapshot:
        try:
            smeta = snap(df, source=src, symbol=sym, start=start, end=end, interval=iv, out_dir=out_dir, timezone="UTC", base_currency=base_currency, fx_source=fx_source, fx_source_ts=fx_source_ts, calendar_id=calendar_id or "", instrument_registry_hash=instrument_registry_hash)
            snapshot_meta = asdict(smeta)
            (out_dir / "snapshot_meta.json").write_text(json.dumps(snapshot_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[snapshot skipped] {e}", file=sys.stderr)

    # 전략 계획 생성
    sma_short, sma_long = 10, 50
    signals_df = sma_cross_long_only(df, short=sma_short, long=sma_long, epsilon=epsilon)
    stops_df = donchian_stop_long(df, N=N)
    sizing_df = build_fixed_fractional_spec(df, N=N, f=f, lot_step=lot_step, V=V, PV=PV)

    # 리밸런싱 스펙 생성 (자산이 2개 이상일 때만 의미 있음)
    num_assets = 1  # 현재는 단일 자산만 지원
    if num_assets > 1:
        close_adj = df[["close_adj"]].rename(columns={"close_adj": sym})
        values_for_rebal = close_adj.shift(1)
        target_weights = pd.Series({sym: 1.0})
        rebal_spec = build_rebalancing_spec_ts(values_for_rebal, target_weights, cash_flow=0.0)
    else:
        empty_df = pd.DataFrame(0.0, index=df.index, columns=[sym])
        rebal_spec = {"buy_notional": empty_df, "sell_notional": empty_df}

    # 엔진 입력을 위한 데이터 재구성
    prices_df = df.copy()
    prices_df.columns = pd.MultiIndex.from_product([[sym], prices_df.columns])
    signals_engine_df = signals_df.to_frame(name=sym)
    stops_engine_df = stops_df[["stop_hit"]].rename(columns={"stop_hit": sym})
    sizing_spec_engine_dict = {sym: sizing_df}

    # 시뮬레이션 엔진 실행
    params_for_meta = {"N": N, "f": f, "epsilon": epsilon, "sma_short": sma_short, "sma_long": sma_long}
    res = eng.run(
        prices=prices_df,
        signals=signals_engine_df,
        stops=stops_engine_df,
        sizing_spec=sizing_spec_engine_dict,
        rebalancing_spec=rebal_spec,
        lot_step=lot_step,
        commission_rate=commission_rate,
        slip=slip,
        initial_equity=initial_equity,
        price_step=price_step,
        base_currency=base_currency,
        snapshot_meta=snapshot_meta,
        params=params_for_meta,
    )

    # 산출물 저장
    pd.DataFrame(res.get("trades", [])).to_csv(out_dir / "trades.csv", index=False)
    pd.DataFrame(res.get("equity_curve", {})).to_csv(out_dir / "equity_curve.csv", index=True)
    
    run_meta = dict(res.get("run_meta", {}))
    if snapshot_meta:
        run_meta.setdefault("snapshot", snapshot_meta)

    (out_dir / "metrics.json").write_text(json.dumps(res.get("metrics", {}), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved to: {out_dir}")
    for v in warns.values():
        print(f"[normalized] {v}")

    return {"out_dir": str(out_dir), "normalized": warns, "files": ["trades.csv", "equity_curve.csv", "metrics.json", "run_meta.json", *([ "snapshot_meta.json"] if snapshot_meta else [])]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """명령줄 인자를 파싱한다."""
    p = argparse.ArgumentParser(description="Backtest pipeline runner")
    p.add_argument("--source", default="yahoo", help="yahoo | upbit")
    p.add_argument("--symbol", required=True, help="e.g., AAPL / 005930.KS / KRW-BTC")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=_today_ymd())
    p.add_argument("--interval", default="1d")

    p.add_argument("--N", type=int, default=20, help="Donchian channel window for stops/sizing")
    p.add_argument("--f", type=float, default=0.02, help="Fractional risk per trade")
    p.add_argument("--lot_step", type=float, default=1.0)
    p.add_argument("--price_step", type=float, default=0.0)
    p.add_argument("--commission_rate", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0005)
    p.add_argument("--epsilon", type=float, default=0.0, help="SMA cross tie-breaker")
    p.add_argument("--initial_equity", type=float, default=1_000_000.0)

    p.add_argument("--V", type=float, help="Futures multiplier (optional)")
    p.add_argument("--PV", type=float, help="FX pip value (optional)")

    p.add_argument("--base_currency", type=str, default="USD")
    p.add_argument("--calendar_id", type=str, default=None)
    p.add_argument("--fx_source", type=str, default=None)
    p.add_argument("--fx_source_ts", type=str, default=None)
    p.add_argument("--instrument_registry_hash", type=str, default=None)

    p.add_argument("--snapshot", action="store_true")
    p.add_argument("--out_dir")
    p.add_argument("--runs_dir", default="runs")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """스크립트의 메인 진입점."""
    args = parse_args(argv)
    root = Path(__file__).resolve().parent
    out_dir = _compose_out_dir(root, args.runs_dir, args.source, args.symbol, args.interval, args.out_dir)
    try:
        run_once(
            source=args.source,
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            interval=args.interval,
            N=args.N,
            f=args.f,
            lot_step=args.lot_step,
            commission_rate=args.commission_rate,
            slip=args.slip,
            epsilon=args.epsilon,
            initial_equity=args.initial_equity,
            V=args.V,
            PV=args.PV,
            snapshot=args.snapshot,
            out_dir=out_dir,
            base_currency=args.base_currency,
            calendar_id=args.calendar_id,
            fx_source=args.fx_source,
            fx_source_ts=args.fx_source_ts,
            instrument_registry_hash=args.instrument_registry_hash,
            price_step=args.price_step,
        )
        return 0
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

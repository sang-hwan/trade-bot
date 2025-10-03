# main.py
from __future__ import annotations

# 표준 라이브러리(우선)
import argparse
import datetime as _dt
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# 서드파티
import pandas as pd

# 프로젝트 모듈
from data.collect import collect
from data.quality_gate import validate as qv
from data.quality_gate import validate_meta as qv_meta
from data.adjust import apply as adj
from data.snapshot import write as snap
import simulation.engine as eng


def _today_ymd() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


def _safe_name(symbol: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in symbol)


def _normalize_for_upbit(symbol: str, interval: str) -> tuple[str, str, dict[str, str]]:
    """Upbit 입력 정규화: 'BTC/KRW'·'BTC-KRW'→'KRW-BTC', '1D/1W/1M'→'day/week/month'."""
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
    if out_dir:
        p = Path(out_dir)
        return p if p.is_absolute() else (root / out_dir)
    # UTC 타임스탬프로 러닝 디렉터리 구성
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
    # 메타/환산 입력 (스냅샷·검증·수집에 사용)
    base_currency: str,
    calendar_id: str | None,
    fx_source: str | None,
    fx_source_ts: str | None,
    instrument_registry_hash: str | None,
    # 라운딩 메타(검증 및 엔진 집행에 적용)
    price_step: float,
) -> dict[str, Any]:
    """수집→정합성→조정→(스냅샷)→엔진→산출물 저장."""
    warns: dict[str, str] = {}
    src = source.lower()
    sym, iv = symbol, interval
    if src == "upbit":
        sym, iv, warns = _normalize_for_upbit(symbol, interval)
        if not calendar_id:
            calendar_id = "24x7"

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 수집
    collected = collect(
        src,
        sym,
        start=start,
        end=end,
        interval=iv,
        base_currency=base_currency,
        calendar_id=calendar_id,
    )
    df = collected.dataframe
    local_ccy = (getattr(collected, "meta", {}) or {}).get("price_currency")

    # 2) 정합성 검증
    qv(df)

    # 3) 완전 조정(*_adj 생성 및 우선 사용)
    df = adj(df)

    # 4) 메타 검증(steps/calendar/FX)
    try:
        qv_meta(
            {
                "base_currency": base_currency,
                "price_currency": local_ccy,
                "calendar_id": calendar_id,
                "lot_step": lot_step,
                "price_step": price_step,
                "fx_source": fx_source,
                "fx_source_ts": fx_source_ts,
            },
            require_fx=bool(local_ccy and base_currency and local_ccy != base_currency),
            require_calendar=True,
            require_steps=price_step > 0.0,
        )
    except Exception as e:
        print(f"[meta warning] {e}", file=sys.stderr)

    # 5) 스냅샷 고정(선택)
    snapshot_meta: dict[str, Any] | None = None
    if snapshot:
        try:
            smeta = snap(
                df,
                source=src,
                symbol=sym,
                start=start,
                end=end,
                interval=iv,
                out_dir=str(out_dir),
                timezone="UTC",
                base_currency=base_currency,
                fx_source=fx_source,
                fx_source_ts=fx_source_ts,
                calendar_id=calendar_id or "",
                instrument_registry_hash=instrument_registry_hash,
            )
            snapshot_meta = asdict(smeta)
            (out_dir / "snapshot_meta.json").write_text(
                json.dumps(snapshot_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            print(f"[snapshot skipped] {e}", file=sys.stderr)

    # 6) 엔진 실행(단일 종목; 기준 통화 환산은 데이터/스냅샷 단계 완료 가정)
    res = eng.run(
        df,
        f=f,
        N=N,
        epsilon=epsilon,
        lot_step=lot_step,
        commission_rate=commission_rate,
        slip=slip,
        V=V,
        PV=PV,
        initial_equity=initial_equity,
        price_step=price_step,
        base_currency=base_currency,
        snapshot_meta=snapshot_meta,
    )

    # 7) 산출물 저장
    trades_obj = res.get("trades", pd.DataFrame())
    equity_obj = res.get("equity_curve", pd.DataFrame())
    trades_df = trades_obj if isinstance(trades_obj, pd.DataFrame) else pd.DataFrame(trades_obj)
    equity_df = equity_obj if isinstance(equity_obj, pd.DataFrame) else pd.DataFrame(equity_obj)

    trades_df.to_csv(out_dir / "trades.csv", index=False)
    equity_df.to_csv(out_dir / "equity_curve.csv", index=True)

    # run_meta 보강(엔진이 이미 기록했으면 setdefault로 보존)
    run_meta = dict(res.get("run_meta", {}))
    if snapshot_meta:
        run_meta.setdefault("snapshot", snapshot_meta)

    (out_dir / "metrics.json").write_text(
        json.dumps(res.get("metrics", {}), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Saved to:", str(out_dir))
    for v in warns.values():
        print("[normalized]", v)

    return {
        "out_dir": str(out_dir),
        "normalized": warns,
        "files": [
            "trades.csv",
            "equity_curve.csv",
            "metrics.json",
            "run_meta.json",
            *(["snapshot_meta.json"] if snapshot_meta else []),
        ],
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest pipeline runner")
    p.add_argument("--source", default="yahoo", help="yahoo | upbit")
    p.add_argument("--symbol", required=True, help="e.g., AAPL / 005930.KS / KRW-BTC / BTC/KRW")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=_today_ymd())
    p.add_argument("--interval", default="1d")

    p.add_argument("--N", type=int, default=20)
    p.add_argument("--f", type=float, default=0.01)
    p.add_argument("--lot_step", type=float, default=1.0)
    p.add_argument("--price_step", type=float, default=0.0)  # 0이면 비활성
    p.add_argument("--commission_rate", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0005)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--initial_equity", type=float, default=1_000_000.0)

    p.add_argument("--V", type=float)
    p.add_argument("--PV", type=float)

    # 스냅샷/메타 입력
    p.add_argument("--base_currency", type=str, default="USD")
    p.add_argument("--calendar_id", type=str, default=None)
    p.add_argument("--fx_source", type=str, default=None)
    p.add_argument("--fx_source_ts", type=str, default=None)
    p.add_argument("--instrument_registry_hash", type=str, default=None)

    p.add_argument("--snapshot", action="store_true")
    p.add_argument("--out_dir")
    p.add_argument("--runs_dir", default="runs")
    return p.parse_args(argv)


def main(argv=None) -> int:
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
        # 종료 코드 반영용 최상위 처리
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

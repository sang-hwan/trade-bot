# main.py
from __future__ import annotations

# ── 표준 라이브러리 (우선 사용)
import argparse
import datetime as _dt
import json
import sys
from dataclasses import asdict
from pathlib import Path

# ── 서드파티
import pandas as pd

# ── 프로젝트 모듈
from data.collect import collect
from data.quality_gate import validate as qv
from data.adjust import apply as adj
from data.snapshot import write as snap
import simulation.engine as eng


def _today_ymd() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


def _safe_name(symbol: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in symbol)


def _normalize_for_upbit(symbol: str, interval: str) -> tuple[str, str, dict[str, str]]:
    """Upbit 입력 정규화: 'BTC/KRW'·'BTC-KRW'→'KRW-BTC', 인터벌 '1d/1w/1M'→'day/week/month'."""
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
    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
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
) -> dict:
    """공개 API: 수집→검증→보정→엔진. 스냅샷은 보조 기능이므로 실패 시 경고만 출력."""
    warns: dict[str, str] = {}
    src = source.lower()
    sym, iv = symbol, interval
    if src == "upbit":
        sym, iv, warns = _normalize_for_upbit(symbol, interval)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 수집(CollectResult) → DataFrame 추출
    collected = collect(src, sym, start, end, iv)
    df = collected.dataframe
    qv(df)
    df = adj(df)

    snapshot_meta = None
    if snapshot:
        try:
            smeta = snap(
                df, source=src, symbol=sym, start=start, end=end, interval=iv,
                out_dir=str(out_dir), timezone="UTC"
            )
            snapshot_meta = asdict(smeta)
            (out_dir / "snapshot_meta.json").write_text(
                json.dumps(snapshot_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            print(f"[snapshot skipped] {e}", file=sys.stderr)

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
    )

    pd.DataFrame(res.get("trades", [])).to_csv(out_dir / "trades.csv", index=False)
    pd.DataFrame(res.get("equity_curve", [])).to_csv(out_dir / "equity_curve.csv", index=False)

    # 스냅샷 메타를 run_meta에 비파괴 병합
    run_meta = dict(res.get("run_meta", {}))
    if snapshot_meta:
        for k, v in snapshot_meta.items():
            run_meta.setdefault(k, v)

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
    p.add_argument("--commission_rate", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0005)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--initial_equity", type=float, default=1_000_000.0)

    p.add_argument("--V", type=float)
    p.add_argument("--PV", type=float)

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
        )
        return 0
    except Exception as e:
        # 최상위 한정: 종료 코드 반영용
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

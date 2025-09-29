from typing import Optional, Dict, Any, Iterable
from pathlib import Path
from datetime import date
from dataclasses import asdict

from collect import collect
from preprocess import preprocess_market_data
from execute import run_backtest
from save import save_run


def backtest(
    *,
    source: str,
    symbol: Optional[str] = None,
    path: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    timezone: Optional[str] = None,
    drop_weekends: bool = False,
    holidays: Optional[Iterable[date]] = None,
    compute_adjusted: bool = True,
    exec_config: Optional[Dict[str, Any]] = None,
    out_root: str = "artifacts",
    run_name: Optional[str] = None,
    version: Optional[str] = None,
    snapshot_dir: Optional[str] = None,
) -> Dict[str, Any]:
    raw = collect(
        source=source,
        symbol=symbol,
        path=path,
        start=start,
        end=end,
        interval=interval,
        timezone=timezone,
        snapshot_dir=snapshot_dir,
    )
    pre = preprocess_market_data(
        raw["df"],
        drop_weekends=drop_weekends,
        holidays=holidays,
        compute_adjusted=compute_adjusted,
        snapshot_dir=snapshot_dir,
        name=f"{(symbol or (Path(path).stem if path else 'data'))}_preprocessed",
    )
    artifacts = run_backtest(pre["df"], cfg=exec_config)
    meta = artifacts.get("meta", {})
    meta["source_meta"] = asdict(raw["meta"]) if "meta" in raw else None
    meta["preprocess_meta"] = asdict(pre["meta"]) if "meta" in pre else None
    artifacts["meta"] = meta
    saved = save_run(artifacts=artifacts, out_root=out_root, run_name=run_name, version=version)
    return {
        "artifacts": {
            "logs": artifacts.get("logs"),
            "equity_curve": artifacts.get("equity_curve"),
            "metrics": artifacts.get("metrics"),
            "meta": artifacts.get("meta"),
        },
        "save": saved,
    }


__all__ = ["backtest"]

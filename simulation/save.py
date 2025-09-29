from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from hashlib import sha256
from datetime import datetime
import json

import pandas as pd


@dataclass
class SaveReport:
    run_dir: str
    logs_path: Optional[str]
    logs_sha256: Optional[str]
    equity_path: Optional[str]
    equity_sha256: Optional[str]
    metrics_path: Optional[str]
    metrics_sha256: Optional[str]
    meta_path: Optional[str]
    meta_sha256: Optional[str]
    manifest_path: str
    manifest_sha256: str


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_parquet(df: Optional[pd.DataFrame], path: Path) -> Optional[Tuple[str, str]]:
    if df is None:
        return None
    df.to_parquet(path, index=True)
    return str(path), _file_sha256(path)


def _write_json(obj: Optional[Dict[str, Any]], path: Path) -> Optional[Tuple[str, str]]:
    if obj is None:
        return None
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(path), _file_sha256(path)


def save_run(
    artifacts: Dict[str, Any],
    out_root: str,
    run_name: Optional[str] = None,
    version: Optional[str] = None,
) -> Dict[str, Any]:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    base = Path(out_root)
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / f"{(run_name or 'run')}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_df: Optional[pd.DataFrame] = artifacts.get("logs")
    equity_df: Optional[pd.DataFrame] = artifacts.get("equity_curve")
    metrics: Optional[Dict[str, Any]] = artifacts.get("metrics")
    meta: Optional[Dict[str, Any]] = artifacts.get("meta")
    logs_info = _write_parquet(logs_df, run_dir / "logs.parquet") if logs_df is not None else None
    equity_info = _write_parquet(equity_df, run_dir / "equity_curve.parquet") if equity_df is not None else None
    metrics_info = _write_json(metrics, run_dir / "metrics.json") if metrics is not None else None
    meta_info = _write_json(meta, run_dir / "meta.json") if meta is not None else None
    files = {}
    if logs_info:
        files["logs.parquet"] = {"path": logs_info[0], "sha256": logs_info[1]}
    if equity_info:
        files["equity_curve.parquet"] = {"path": equity_info[0], "sha256": equity_info[1]}
    if metrics_info:
        files["metrics.json"] = {"path": metrics_info[0], "sha256": metrics_info[1]}
    if meta_info:
        files["meta.json"] = {"path": meta_info[0], "sha256": meta_info[1]}
    manifest = {
        "run_dir": str(run_dir),
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "version": version,
        "files": files,
    }
    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    manifest_sha = _file_sha256(manifest_path)
    report = SaveReport(
        run_dir=str(run_dir),
        logs_path=files.get("logs.parquet", {}).get("path"),
        logs_sha256=files.get("logs.parquet", {}).get("sha256"),
        equity_path=files.get("equity_curve.parquet", {}).get("path"),
        equity_sha256=files.get("equity_curve.parquet", {}).get("sha256"),
        metrics_path=files.get("metrics.json", {}).get("path"),
        metrics_sha256=files.get("metrics.json", {}).get("sha256"),
        meta_path=files.get("meta.json", {}).get("path"),
        meta_sha256=files.get("meta.json", {}).get("sha256"),
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha,
    )
    return {"report": report, "manifest": manifest}


__all__ = ["SaveReport", "save_run"]

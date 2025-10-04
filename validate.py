# validate.py
"""
Validate backtest artifacts and produce reports.

계약:
- 입력: RUN_DIR (trades.csv, equity_curve.csv, metrics.json, run_meta.json, snapshot_meta.json 포함)
- 동작: data/strategy/simulation 게이트 실행 + 분석/시각화 산출
- 출력:
  - RUN_DIR/validation/gate_summary.json (모든 게이트 결과)
  - RUN_DIR/validation/fig_*.png, analysis_summary.json (analysis_viz 생성물)
- 종료코드:
  - 0: 모든 게이트 통과
  - 1: 하나 이상 실패(검증 위반)
  - 2: 입력/IO 오류 등 준비 실패
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import pandas as pd

from validation import analysis_viz, data_gate, simulation_gate, strategy_gate


_REQUIRED_FILES = {
    "trades": "trades.csv",
    "equity_curve": "equity_curve.csv",
    "metrics": "metrics.json",
    "run_meta": "run_meta.json",
    "snapshot_meta": "snapshot_meta.json",
}


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"파일이 없습니다: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱 실패: {path} ({e})") from e
    except OSError as e:
        raise OSError(f"파일 읽기 실패: {path} ({e})") from e


def _ensure_required(run_dir: str) -> Dict[str, str]:
    missing = []
    paths: Dict[str, str] = {}
    for k, fn in _REQUIRED_FILES.items():
        p = os.path.join(run_dir, fn)
        if not os.path.isfile(p):
            missing.append(fn)
        paths[k] = p
    if missing:
        raise FileNotFoundError(f"RUN_DIR에 필수 파일이 없습니다: {', '.join(missing)}")
    return paths


def _load_artifacts(paths: Dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any], str]:
    try:
        trades = pd.read_csv(paths["trades"])
    except Exception as e:
        raise ValueError(f"trades.csv 읽기 실패: {e}") from e

    try:
        equity_curve = pd.read_csv(paths["equity_curve"], index_col=0, parse_dates=True)
    except Exception as e:
        raise ValueError(f"equity_curve.csv 읽기 실패: {e}") from e

    metrics = _read_json(paths["metrics"])
    run_meta = _read_json(paths["run_meta"])
    snapshot_meta = _read_json(paths["snapshot_meta"])

    snapshot_parquet_path = snapshot_meta.get("snapshot_path")
    if not snapshot_parquet_path:
        # data_gate가 상세히 잡아주지만, 여기서도 명확히 실패 처리
        raise ValueError("snapshot_meta.json에 snapshot_path가 없습니다.")
    if not os.path.isabs(snapshot_parquet_path):
        snapshot_parquet_path = os.path.join(os.path.dirname(paths["snapshot_meta"]), snapshot_parquet_path)

    return trades, equity_curve, metrics, run_meta, snapshot_meta, snapshot_parquet_path


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate backtest run artifacts and produce reports.")
    ap.add_argument("run_dir", help="Backtest run directory (contains CSV/JSON artifacts).")
    ap.add_argument("--docs-dir", default=None, help="Docs directory for display-math lint (optional).")
    args = ap.parse_args(argv)

    run_dir = args.run_dir

    # 1) 필수 파일 확인 및 로드
    try:
        paths = _ensure_required(run_dir)
        trades, equity_curve, metrics, run_meta, snapshot_meta, snapshot_parquet_path = _load_artifacts(paths)
    except (FileNotFoundError, ValueError, OSError) as e:
        # 준비 실패 → 종료코드 2
        print(json.dumps({"passed": False, "error": str(e)}, ensure_ascii=False))
        return 2

    # 2) 데이터 게이트
    dg_res = data_gate.run(
        data_gate.Artifacts(
            snapshot_parquet_path=snapshot_parquet_path,
            snapshot_meta=snapshot_meta,
            docs_dir=args.docs_dir,
        )
    )

    # 3) 전략 게이트
    sg_res = strategy_gate.run(
        strategy_gate.Artifacts(
            snapshot_parquet_path=snapshot_parquet_path,
            run_meta=run_meta,
            trades=trades,
        )
    )

    # 4) 시뮬레이션 게이트
    sim_res = simulation_gate.run(
        simulation_gate.Artifacts(
            snapshot_parquet_path=snapshot_parquet_path,
            run_meta=run_meta,
            snapshot_meta=snapshot_meta,
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
        )
    )

    # 5) 분석/시각화
    viz_out_dir = os.path.join(run_dir, "validation")
    vz_res = analysis_viz.run(
        analysis_viz.Artifacts(
            out_dir=viz_out_dir,
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            run_meta=run_meta,
        )
    )

    # 6) 요약 저장 및 종료코드
    all_res = {
        "data_gate": dg_res,
        "strategy_gate": sg_res,
        "simulation_gate": sim_res,
        "analysis_viz": vz_res,
    }
    _write_json(os.path.join(viz_out_dir, "gate_summary.json"), all_res)

    passed = dg_res["passed"] and sg_res["passed"] and sim_res["passed"] and vz_res["passed"]
    print(json.dumps({"passed": bool(passed)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

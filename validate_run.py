# validate_run.py
"""
검증 파이프라인(권장 경로): 워크포워드 → 그리드 → 평가 → 리포트.

계약
- 입력: --run_dir 하위 run_meta.json 또는 snapshot_meta.json의 Parquet 스냅샷 경로 사용.
- 출력: --out_dir 아래 히트맵/산점도/요약MD/자본곡선 PNG 및 paths.json 생성.
- 실패: 필수 파일/키 누락, 형식 위반, equity 누락 등은 ValueError로 즉시 실패.
"""

from __future__ import annotations

import argparse
import json
import inspect
from pathlib import Path

import pandas as pd

from validation.splits import walk_forward
from validation.grid import grid_from_spaces, linspace
from validation.evaluate import evaluate
from validation.report import render_report
from simulation.engine import run as run_engine


def _load_snapshot_path(run_dir: Path) -> Path:
    """run_meta.json/snapshot_meta.json에서 스냅샷 경로 추출(없으면 실패)."""
    run_meta_p = run_dir / "run_meta.json"
    snap_meta_p = run_dir / "snapshot_meta.json"
    if run_meta_p.exists():
        snap_path = json.loads(run_meta_p.read_text(encoding="utf-8")).get("snapshot_path")
        if snap_path:
            return Path(snap_path)
    if snap_meta_p.exists():
        return Path(json.loads(snap_meta_p.read_text(encoding="utf-8"))["snapshot_path"])
    raise ValueError("snapshot_path not found in run_meta.json or snapshot_meta.json.")


def _select_combo_ids(agg: pd.DataFrame, robust: pd.DataFrame, *, top_k_curves: int) -> list[int]:
    """자본곡선 대상 combo_id 선정: 로버스트 상위 → median 상위 보충."""
    if "combo_id" not in agg.columns or "median" not in agg.columns:
        raise ValueError("Aggregated results must contain 'combo_id' and 'median'.")
    ids: list[int] = []
    if robust is not None and not robust.empty:
        ids += [int(i) for i in robust.sort_values("median", ascending=False, kind="mergesort")["combo_id"].head(top_k_curves)]
    if len(ids) < top_k_curves:
        rest = (
            agg[~agg["combo_id"].isin(ids)]
            .sort_values("median", ascending=False, kind="mergesort")["combo_id"]
            .head(top_k_curves - len(ids))
        )
        ids += [int(i) for i in rest]
    return ids


def _collect_equities(df: pd.DataFrame, agg: pd.DataFrame, combo_ids: list[int]) -> dict[int, pd.DataFrame]:
    """선정 combo_id에 대해 엔진을 재실행하고 equity 시계열 수집."""
    equities: dict[int, pd.DataFrame] = {}
    sig = inspect.signature(run_engine)
    allowed = {name for name, p in sig.parameters.items()
               if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)}
    allowed.discard("df")
    required = {name for name, p in sig.parameters.items()
                if name != "df"
                and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is inspect._empty}
    
    for cid in combo_ids:
        row = agg.loc[agg["combo_id"] == cid]
        if row.empty:
            raise ValueError(f"combo_id {cid} not found in aggregated results.")
        raw_params = row["params"].iloc[0]
        params = {k: v for k, v in raw_params.items() if k in allowed}
        missing = sorted(required - params.keys())
        if missing:
            raise ValueError(f"Missing required engine params for combo_id {cid}: {missing}")
        try:
            outputs = run_engine(df, **params)
        except TypeError as e:
            raise ValueError(f"engine.run failed for combo_id {cid} with params {params}: {e}") from e
        
        eq = outputs["equity_curve"] if isinstance(outputs, dict) else getattr(outputs, "equity_curve", None)
        if not isinstance(eq, pd.DataFrame):
            raise ValueError("engine outputs must include DataFrame 'equity_curve'.")
        if "ts" not in eq.columns:
            eq = eq.reset_index().rename(columns={"index": "ts"})
        eq["ts"] = pd.to_datetime(eq["ts"], utc=True)
        equities[cid] = eq[["ts", "equity"]]
    return equities
        
def main() -> None:
    ap = argparse.ArgumentParser(description="Validation pipeline (recommended).")
    ap.add_argument("--run_dir", required=True, help="Directory containing main.py outputs.")
    ap.add_argument("--out_dir", default="validation_report", help="Output directory for report assets.")
    ap.add_argument("--train_years", type=int, default=3)
    ap.add_argument("--test_years", type=int, default=1)
    ap.add_argument("--step_years", type=int, default=1)
    ap.add_argument("--top_k", type=int, default=20, help="robust 상위 N")
    ap.add_argument("--top_k_curves", type=int, default=5, help="자본곡선 PNG 생성 수")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 스냅샷 로드
    snap_path = _load_snapshot_path(run_dir)
    df = pd.read_parquet(snap_path)

    # 2) 워크포워드 분할
    splits = walk_forward(
        df,
        train_years=args.train_years,
        test_years=args.test_years,
        step_years=args.step_years,
        mode="rolling",
        allow_incomplete_last=False,
    )

    # 3) 파라미터 그리드(권장 범위)
    spaces = {
        "sma_short": [10, 20, 30],
        "sma_long": [50, 100, 150],
        "N": [10, 20, 40],
        "f": linspace(0.005, 0.03, 6, round_to=4),
        "commission_rate": [0.0005, 0.001, 0.002],
        "slip": [0.0, 0.0005, 0.001],
        "epsilon": [0.0],
        "lot_step": [0.01],
    }
    combos = grid_from_spaces(spaces, max_size=200_000, validate=True)

    # 4) 평가(스플릿 기준)
    results = evaluate(
        snapshot_df=df,
        combos=combos,
        splits=splits,
        metric_key="total_return",
        risk_key="mdd",
        min_worst=-0.20,
        max_iqr=0.10,
        top_k=args.top_k,
    )
    if "agg" not in results or "robust" not in results:
        raise ValueError("Evaluation results must contain 'agg' and 'robust' DataFrames.")
    agg = results["agg"]
    robust = results["robust"]

    # 5) 리포트(자본곡선 수집 → 렌더링)
    combo_ids = _select_combo_ids(agg, robust, top_k_curves=args.top_k_curves)
    equities = _collect_equities(df, agg, combo_ids)
    paths = render_report(
        results=results,
        out_dir=str(out_dir),
        equities=equities,
        x_key=None,
        y_key=None,
        metric_key="median",
        risk_worst_key="mdd_worst",
        iqr_key="iqr",
        top_k_curves=args.top_k_curves,
    )
    (out_dir / "paths.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

# validation/report.py
"""
리포팅/시각화(자본곡선, 히트맵, 베스트 vs 로버스트 하이라이트).

공개 API
- render_report(results: dict[str, pd.DataFrame], out_dir: str, *,
                equities: dict[int, pd.DataFrame],
                x_key: str | None = None, y_key: str | None = None,
                metric_key: str = "median", risk_worst_key: str = "mdd_worst",
                iqr_key: str = "iqr", top_k_curves: int = 5) -> dict[str, str]

계약
- results["agg"]는 최소 ["combo_id", metric_key, iqr_key, risk_worst_key, "params"] 포함.
- results["robust"]가 있으면 로버스트 표시에 사용(없으면 빈 DataFrame 처리).
- equities는 {combo_id: DataFrame("ts","equity")} (UTC) 형식이어야 함.
- 필수 키/형식 위반 시 ValueError.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ("render_report",)


# ---------- 내부 유틸 ----------

def _ensure_out_dir(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _expand_params(df: pd.DataFrame) -> pd.DataFrame:
    """'params'(dict) 열을 컬럼으로 확장."""
    if "params" not in df.columns:
        raise ValueError("results['agg'] must contain a 'params' column.")
    params_df = pd.json_normalize(df["params"])
    params_df.columns = [str(c) for c in params_df.columns]
    return pd.concat([df.drop(columns=["params"]), params_df], axis=1)


def _auto_axes_keys(agg_params: pd.DataFrame, *, exclude: set[str]) -> tuple[str, str]:
    """숫자형 파라미터 중 x/y 자동 선택(결정적 순서)."""
    candidates = [
        c for c in agg_params.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(agg_params[c])
    ]
    if len(candidates) < 2:
        raise ValueError("Not enough numeric parameter columns for a heatmap; set x_key/y_key explicitly.")
    return candidates[0], candidates[1]


def _save_fig(path: Path) -> None:
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------- 플롯 ----------

def _plot_equity_curve(equity: pd.DataFrame, title: str, out_path: Path) -> None:
    """자본곡선 단일 플롯."""
    if not {"ts", "equity"}.issubset(equity.columns):
        raise ValueError("equities[combo_id] must contain 'ts' and 'equity'.")
    df = equity.sort_values("ts").copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    plt.figure()
    plt.plot(df["ts"], df["equity"])
    plt.title(title)
    plt.xlabel("time (UTC)")
    plt.ylabel("equity")
    plt.grid(True)
    _save_fig(out_path)


def _plot_heatmap(agg_params: pd.DataFrame, *, x_key: str, y_key: str, z_key: str,
                  out_path: Path, title: str) -> None:
    """파라미터 히트맵(피벗→imshow)."""
    for k in (x_key, y_key, z_key):
        if k not in agg_params.columns:
            raise ValueError(f"'{k}' not found in aggregated DataFrame.")
    pivot = agg_params.pivot_table(index=y_key, columns=x_key, values=z_key, aggfunc="median")
    pivot = pivot.sort_index().sort_index(axis=1)

    plt.figure()
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.xticks(range(pivot.shape[1]), pivot.columns.astype(str), rotation=45, ha="right")
    plt.yticks(range(pivot.shape[0]), pivot.index.astype(str))
    plt.colorbar()
    _save_fig(out_path)


def _plot_scatter_best_vs_robust(agg: pd.DataFrame, robust: pd.DataFrame, *,
                                 metric_key: str, iqr_key: str,
                                 out_path: Path, title: str) -> None:
    """베스트 vs 로버스트 산점도: x=IQR, y=median."""
    plt.figure()
    plt.scatter(agg[iqr_key], agg[metric_key], label="all")
    if not robust.empty:
        inter = agg.merge(robust[["combo_id"]], on="combo_id", how="inner")
        plt.scatter(inter[iqr_key], inter[metric_key], marker="x", label="robust")
    plt.title(title)
    plt.xlabel(iqr_key)
    plt.ylabel(metric_key)
    plt.legend()
    plt.grid(True)
    _save_fig(out_path)


# ---------- 요약 파일 ----------

def _write_summary_md(agg: pd.DataFrame, robust: pd.DataFrame, out_path: Path, *,
                      metric_key: str, iqr_key: str, risk_worst_key: str, top_n: int = 10) -> None:
    """요약 MD: median 상위, 로버스트 목록."""
    lines: list[str] = ["# Report Summary\n", "## Top by median\n"]
    top_med = agg.sort_values(metric_key, ascending=False).head(top_n)
    for _, row in top_med.iterrows():
        lines.append(
            f"- combo_id={int(row['combo_id'])}, "
            f"{metric_key}={row[metric_key]:.4f}, {iqr_key}={row[iqr_key]:.4f}, "
            f"{risk_worst_key}={row[risk_worst_key]:.4f}"
        )
    lines.append("\n## Robust (filtered)\n")
    if robust.empty:
        lines.append("- (no robust combos)")
    else:
        rob = robust.sort_values(metric_key, ascending=False)
        for _, row in rob.iterrows():
            lines.append(
                f"- combo_id={int(row['combo_id'])}, "
                f"{metric_key}={row[metric_key]:.4f}, {iqr_key}={row[iqr_key]:.4f}, "
                f"{risk_worst_key}={row[risk_worst_key]:.4f}"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- 메인 오케스트레이션 ----------

def render_report(
    results: dict[str, pd.DataFrame],
    out_dir: str,
    *,
    equities: dict[int, pd.DataFrame],
    x_key: str | None = None,
    y_key: str | None = None,
    metric_key: str = "median",
    risk_worst_key: str = "mdd_worst",
    iqr_key: str = "iqr",
    top_k_curves: int = 5,
) -> dict[str, str]:
    """
    리포트 생성:
      - heatmap_*.png, scatter_best_vs_robust.png, summary.md
      - equity_combo_*.png (로버스트 상위 → median 상위, 최대 top_k_curves)
    """
    if "agg" not in results or not isinstance(results["agg"], pd.DataFrame):
        raise ValueError("results must contain an 'agg' DataFrame.")
    agg = results["agg"].copy()
    robust = results.get("robust", pd.DataFrame()).copy()

    required_cols = {"combo_id", metric_key, iqr_key, risk_worst_key, "params"}
    if not required_cols.issubset(agg.columns):
        missing = required_cols - set(agg.columns)
        raise ValueError(f"results['agg'] is missing required columns: {sorted(missing)}")
    if not isinstance(equities, dict) or not equities:
        raise ValueError("equities must be a non-empty dict of {combo_id: DataFrame('ts','equity')}.")

    out = _ensure_out_dir(out_dir)
    outputs: dict[str, str] = {}

    # params 확장 및 히트맵 축 결정
    agg_params = _expand_params(agg)
    exclude = {"combo_id", metric_key, iqr_key, risk_worst_key}
    if x_key is None or y_key is None:
        x_auto, y_auto = _auto_axes_keys(agg_params, exclude=exclude)
        x_key = x_key or x_auto
        y_key = y_key or y_auto

    # 히트맵
    heatmap_path = out / f"heatmap_{x_key}_vs_{y_key}_{metric_key}.png"
    _plot_heatmap(agg_params, x_key=x_key, y_key=y_key, z_key=metric_key,
                  out_path=heatmap_path, title=f"Heatmap: {metric_key}")
    outputs["heatmap"] = str(heatmap_path)

    # 베스트 vs 로버스트 산점도
    scatter_path = out / "scatter_best_vs_robust.png"
    _plot_scatter_best_vs_robust(agg=agg, robust=robust, metric_key=metric_key,
                                 iqr_key=iqr_key, out_path=scatter_path,
                                 title="Best vs Robust (x=IQR, y=median)")
    outputs["scatter_best_vs_robust"] = str(scatter_path)

    # 요약 MD
    summary_path = out / "summary.md"
    _write_summary_md(agg=agg, robust=robust, out_path=summary_path,
                      metric_key=metric_key, iqr_key=iqr_key, risk_worst_key=risk_worst_key)
    outputs["summary_md"] = str(summary_path)

    # 자본곡선: 로버스트 상위 → median 상위(안정 정렬로 결정성 유지)
    target_ids: list[int] = []
    if not robust.empty:
        target_ids.extend(
            [int(i) for i in robust.sort_values(metric_key, ascending=False, kind="mergesort")["combo_id"].head(top_k_curves)]
        )
    if len(target_ids) < top_k_curves:
        rest = (
            agg[~agg["combo_id"].isin(target_ids)]
            .sort_values(metric_key, ascending=False, kind="mergesort")["combo_id"]
            .head(top_k_curves - len(target_ids))
        )
        target_ids.extend([int(i) for i in rest])

    # 대상 모두 존재해야 함(엄격)
    missing_eq = [cid for cid in target_ids if cid not in equities]
    if missing_eq:
        raise ValueError(f"equities is missing equity curves for combo_id(s): {missing_eq}")

    for cid in target_ids:
        eq_path = out / f"equity_combo_{cid}.png"
        _plot_equity_curve(equities[cid], title=f"Equity Curve — combo_id={cid}", out_path=eq_path)
        outputs[f"equity_combo_{cid}"] = str(eq_path)

    return outputs

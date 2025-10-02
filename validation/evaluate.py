# validation/evaluate.py
"""
조합별 시뮬레이션 실행 및 metrics 집계(IQR·최악 포함 로버스트 식별).

공개 API
- evaluate(snapshot_df, combos, *, splits=None, run_fn=None,
           metric_key="total_return", risk_key="mdd",
           min_worst=-0.20, max_iqr=0.10, top_k=20) -> dict[str, pd.DataFrame]
  반환: {"per_split": DF, "agg": DF, "robust": DF}

계약
- snapshot_df: DatetimeIndex(tz='UTC') 단조 증가·중복 없음(quality_gate 단계 보장).
- combos: 엔진 run_fn(snapshot_df, params)로 실행할 파라미터 dict 리스트.
- splits: 없으면 전체 구간 1개("ALL"). 있으면 각 split의 test 구간을 집계 기준으로 사용.
- 로버스트: metric IQR ≤ max_iqr, worst(최소값) ≥ min_worst 통과 조합 중 median 내림차순 상위 top_k.
"""

from __future__ import annotations

from typing import Callable, Iterable, Any
import inspect
import pandas as pd

__all__ = ["evaluate"]


# ---- 내부 유틸 ----

def _as_index_utc(df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("snapshot_df must have a DatetimeIndex.")
    if idx.tz is None:
        raise ValueError("snapshot_df index must be tz-aware (UTC).")
    tzname = getattr(idx.tz, "key", getattr(idx.tz, "zone", str(idx.tz)))
    if tzname != "UTC":
        raise ValueError("snapshot_df index timezone must be 'UTC'.")
    if not idx.is_monotonic_increasing:
        raise ValueError("snapshot_df index must be strictly increasing.")
    if idx.has_duplicates:
        raise ValueError("snapshot_df index must not contain duplicates.")
    return idx


def _extract_outputs(outputs: Any) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    엔진 반환 표준화:
    dict형 {"trades","equity_curve","metrics","run_meta"} 또는
    속성형(.trades, .equity_curve, .metrics, .run_meta) 모두 허용.
    """
    if isinstance(outputs, dict):
        trades = outputs.get("trades")
        equity = outputs.get("equity_curve")
        metrics = outputs.get("metrics", {}) or {}
        meta = outputs.get("run_meta", {}) or {}
    else:
        trades = getattr(outputs, "trades")
        equity = getattr(outputs, "equity_curve")
        metrics = getattr(outputs, "metrics", {}) or {}
        meta = getattr(outputs, "run_meta", {}) or {}
    if not isinstance(trades, pd.DataFrame) or not isinstance(equity, pd.DataFrame):
        raise TypeError("engine outputs must include DataFrame 'trades' and 'equity_curve'.")

    # equity 표준화: 'ts' 타임스탬프 컬럼 보장(UTC)
    if "ts" not in equity.columns or "equity" not in equity.columns:
        if isinstance(equity.index, pd.DatetimeIndex) and "equity" in equity.columns:
            equity = equity.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("equity_curve must contain 'ts' and 'equity' columns or have a DatetimeIndex + 'equity' column.")
    equity["ts"] = pd.to_datetime(equity["ts"], utc=True)
    return trades, equity, metrics, meta


def _window_metrics(equity: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """
    구간 재기준화 메트릭:
    - total_return: 구간 시작 대비 마지막 자본 - 1
    - mdd: 구간 내 상대자본(rebased) 대비 최소 드로다운
    """
    sub = equity[(equity["ts"] >= start) & (equity["ts"] <= end)]
    if sub.empty:
        return {"total_return": float("nan"), "mdd": float("nan")}
    sub = sub.sort_values("ts")
    e0 = float(sub["equity"].iloc[0])
    if e0 <= 0:
        return {"total_return": float("nan"), "mdd": float("nan")}
    rel = sub["equity"] / e0
    tr = float(rel.iloc[-1] - 1.0)
    mdd = float((rel / rel.cummax() - 1.0).min())
    return {"total_return": tr, "mdd": mdd}


def _ensure_run_fn(run_fn: Callable | None) -> Callable:
    if run_fn is not None:
        return run_fn
    # 지연 임포트(순환참조 회피)
    from simulation import engine as _engine  # type: ignore
    return _engine.run

def _filter_kwargs(func: Callable, params: dict) -> dict:
    """엔진 시그니처 기반으로 허용된 키만 전달."""
    sig = inspect.signature(func)
    allowed = {name for name, p in sig.parameters.items()
               if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)}
    allowed.discard("df")  # 첫 인자는 데이터프레임
    return {k: v for k, v in params.items() if k in allowed}

# ---- 평가 본체 ----

def evaluate(
    snapshot_df: pd.DataFrame,
    combos: list[dict],
    *,
    splits: Iterable | None = None,
    run_fn: Callable[[pd.DataFrame, dict], Any] | None = None,
    metric_key: str = "total_return",
    risk_key: str = "mdd",
    min_worst: float = -0.20,
    max_iqr: float = 0.10,
    top_k: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    조합별 실행 → 스플릿별 메트릭 → 집계/로버스트 식별.
    반환:
      - per_split: combo_id, split_id, {metric_key,risk_key}, params
      - agg: combo_id별 median/q25/q75/iqr/worst/best/n + params
      - robust: 로버스트 필터 통과 상위 top_k(median desc)
    """
    _as_index_utc(snapshot_df)
    run = _ensure_run_fn(run_fn)

    per_rows: list[dict] = []
    for combo_idx, params in enumerate(combos, start=1):
        outputs = run(snapshot_df, **_filter_kwargs(run, params))
        _, equity, _, _ = _extract_outputs(outputs)

        if splits:
            for split_idx, split in enumerate(splits, start=1):
                start = getattr(split, "test_start", None) or getattr(split, "train_start", None)
                end = getattr(split, "test_end", None) or getattr(split, "train_end", None)
                if start is None or end is None:
                    raise ValueError("split must provide test_start/test_end or train_start/train_end.")
                wm = _window_metrics(equity, start=start, end=end)
                per_rows.append({
                    "combo_id": combo_idx,
                    "split_id": split_idx,
                    metric_key: wm.get(metric_key, float("nan")),
                    risk_key: wm.get(risk_key, float("nan")),
                    "params": params,
                })
        else:
            start, end = equity["ts"].min(), equity["ts"].max()
            wm = _window_metrics(equity, start=start, end=end)
            per_rows.append({
                "combo_id": combo_idx,
                "split_id": 1,
                metric_key: wm.get(metric_key, float("nan")),
                risk_key: wm.get(risk_key, float("nan")),
                "params": params,
            })

    per_split = pd.DataFrame(per_rows)
    if per_split.empty:
        return {"per_split": per_split, "agg": pd.DataFrame(), "robust": pd.DataFrame()}

    # 집계: 중앙값, 사분위, IQR, 최악/최고, 관측 수
    def _agg(df: pd.DataFrame, key: str) -> pd.DataFrame:
        return df.groupby("combo_id", as_index=False)[key].agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            iqr=lambda s: s.quantile(0.75) - s.quantile(0.25),
            worst="min",
            best="max",
            n="count",
        )

    agg_m = _agg(per_split, metric_key)
    agg_r = _agg(per_split, risk_key).rename(columns={
        "median": f"{risk_key}_median",
        "q25": f"{risk_key}_q25",
        "q75": f"{risk_key}_q75",
        "iqr": f"{risk_key}_iqr",
        "worst": f"{risk_key}_worst",
        "best": f"{risk_key}_best",
        "n": f"{risk_key}_n",
    })
    agg = pd.merge(agg_m, agg_r, on="combo_id", how="inner")

    # params(대표값) 병합
    params_map = per_split.groupby("combo_id", as_index=False)["params"].first()
    agg = pd.merge(agg, params_map, on="combo_id", how="left")

    # 로버스트 필터
    robust = agg[(agg["iqr"] <= max_iqr) & (agg[f"{risk_key}_worst"] >= min_worst)].copy()
    robust = robust.sort_values("median", ascending=False).head(top_k).reset_index(drop=True)

    return {
        "per_split": per_split.sort_values(["combo_id", "split_id"]).reset_index(drop=True),
        "agg": agg.sort_values("combo_id").reset_index(drop=True),
        "robust": robust,
    }

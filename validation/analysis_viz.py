# validation/analysis_viz.py
"""
Analysis & Visualization

의도:
- 결과를 한눈에 파악할 수 있도록 핵심 지표와 시각화를 생성한다.
  * 자본곡선, 드로우다운, 롤링 샤프(일봉 252), 거래별 실현손익 분포
  * 요약 지표(JSON)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict
import json
import math
import os

import pandas as pd
import matplotlib.pyplot as plt


class VizResult(TypedDict):
    passed: bool
    errors: List[str]
    warnings: List[str]
    evidence: Dict[str, Any]


@dataclass
class Artifacts:
    """입력 산출물 핸들."""
    out_dir: str
    equity_curve: pd.DataFrame         # index: UTC DatetimeIndex, columns: ["equity"]
    trades: pd.DataFrame               # columns: ["ts","side","reason","qty","price","commission","slip","realized_pnl",...]
    metrics: Dict[str, Any]
    run_meta: Dict[str, Any]


# ---------- 유틸 ----------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _rolling_sharpe(ec: pd.Series, window: int = 126, periods_per_year: int = 252) -> pd.Series:
    """일간 수익률 기반 롤링 샤프(연환산)."""
    ret = ec.pct_change().replace([math.inf, -math.inf], pd.NA).dropna()
    mu = ret.rolling(window, min_periods=window).mean()
    sd = ret.rolling(window, min_periods=window).std(ddof=0)
    return (mu / sd) * math.sqrt(periods_per_year)


def _cagr(ec: pd.Series) -> float:
    if ec.empty or ec.iloc[0] <= 0:
        return float("nan")
    days = (ec.index[-1] - ec.index[0]).days
    if days <= 0:
        return float("nan")
    years = days / 365.25
    return float((ec.iloc[-1] / ec.iloc[0]) ** (1.0 / years) - 1.0)


def _save_png(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------- 본체 ----------

def run(artifacts: Artifacts) -> VizResult:
    """결과 분석/시각화 단일 진입점."""
    errors: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {}

    # 입력 확인
    if artifacts.equity_curve is None or artifacts.equity_curve.empty or "equity" not in artifacts.equity_curve.columns:
        return VizResult(passed=False, errors=["[equity_curve] 'equity' 시계열이 필요합니다."], warnings=[], evidence={})
    if artifacts.trades is None:
        return VizResult(passed=False, errors=["[trades] DataFrame이 필요합니다."], warnings=[], evidence={})

    _ensure_dir(artifacts.out_dir)

    # 시계열 정규화
    ec = artifacts.equity_curve.copy()
    ec.index = pd.to_datetime(ec.index, utc=True)
    equity = _safe_series(ec["equity"]).dropna()

    # 드로우다운/샤프
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    rs = _rolling_sharpe(equity, window=126, periods_per_year=252)

    # 요약 지표
    init_eq = float(artifacts.metrics.get("initial_equity", float("nan")))
    final_eq = float(artifacts.metrics.get("final_equity", float("nan")))
    total_return = float(artifacts.metrics.get("total_return", float("nan")))
    mdd = float(artifacts.metrics.get("mdd", float("nan")))
    cagr = _cagr(equity)

    trades = artifacts.trades.copy()
    if "ts" in trades.columns:
        trades["ts"] = pd.to_datetime(trades["ts"], utc=True)
        trades.sort_values(["ts"], kind="stable", inplace=True)
    realized = _safe_series(trades.get("realized_pnl", pd.Series(dtype="float64"))).dropna()
    n_trades = int(len(trades))
    wins = realized[realized > 0]
    losses = realized[realized < 0]
    win_rate = float(len(wins) / max(1, len(realized))) if len(realized) else float("nan")
    payoff = float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else float("nan")

    # 제목 표시용 기준 통화
    base_ccy = str(artifacts.run_meta.get("base_currency", "")) or str((artifacts.run_meta.get("snapshot") or {}).get("base_currency", "") or "")
    title_suffix = f" [{base_ccy}]" if base_ccy else ""

    # 그림 생성
    path_eq = os.path.join(artifacts.out_dir, "fig_equity.png")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(equity.index, equity.values)
    ax1.set_title(f"Equity Curve{title_suffix}")
    ax1.set_xlabel("Time (UTC)")
    ax1.set_ylabel("Equity")
    _save_png(fig1, path_eq)

    path_dd = os.path.join(artifacts.out_dir, "fig_drawdown.png")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(dd.index, dd.values)
    ax2.set_title("Drawdown")
    ax2.set_xlabel("Time (UTC)")
    ax2.set_ylabel("Drawdown")
    _save_png(fig2, path_dd)

    path_rs = os.path.join(artifacts.out_dir, "fig_rolling_sharpe.png")
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.plot(rs.index, rs.values)
    ax3.set_title("Rolling Sharpe (126-day, ann.252)")
    ax3.set_xlabel("Time (UTC)")
    ax3.set_ylabel("Sharpe")
    _save_png(fig3, path_rs)

    path_pnl = os.path.join(artifacts.out_dir, "fig_trade_pnl.png")
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    bins = min(60, max(10, int(math.sqrt(max(1, len(realized))))))
    ax4.hist(realized.values, bins=bins)
    ax4.set_title("Trade Realized PnL Distribution")
    ax4.set_xlabel("Realized PnL")
    ax4.set_ylabel("Count")
    _save_png(fig4, path_pnl)

    # 요약 JSON
    summary = {
        "initial_equity": init_eq,
        "final_equity": final_eq,
        "total_return": total_return,
        "mdd": mdd,
        "cagr": cagr,
        "n_trades": n_trades,
        "n_realized": int(len(realized)),
        "win_rate": win_rate,
        "payoff": payoff,
        "paths": {
            "equity_png": path_eq,
            "drawdown_png": path_dd,
            "rolling_sharpe_png": path_rs,
            "trade_pnl_png": path_pnl,
        },
    }
    summary_path = os.path.join(artifacts.out_dir, "analysis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    evidence.update(
        {
            "summary_path": summary_path,
            "figure_paths": [path_eq, path_dd, path_rs, path_pnl],
            "stats": {"cagr": cagr, "win_rate": win_rate, "payoff": payoff},
        }
    )

    return VizResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
    )

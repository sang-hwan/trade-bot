# strategy/rebalancing_spec.py
"""
현금흐름 리밸런싱(Cash-Flow Rebalancing) 스펙 빌더.

공개 API
- build_rebalancing_spec_ts(values, target_weights, cash_flow) -> dict[str, Any]
- build_rebalancing_spec_t(values_t, target_weights, cash_flow_t, *, at=None) -> dict[str, Any]

계약(기준 통화 기준)
- values: 각 시점(t-1)의 자산별 평가액 V_i(t-1) [DataFrame: index=time, columns=assets]
- target_weights: 목표 비중 w_i (행합=1) [Series(index=assets) 또는 DataFrame(index=time, columns=assets)]
- cash_flow: 순현금흐름 F_t (유입≥0, 유출<0) [float 또는 Series(index=time)]

규칙(요약)
- P_{t-1}=ΣV_i, P*=P_{t-1}+F_t, T_i=w_i·P*, d_i=T_i−V_i
- 유입(F_t≥0): 미달 자산(I^+={d_i>0})에 비례 매수 Δ_i^buy=(d_i/S^+)·min(F_t, S^+), S^+=Σ_{I^+}d_i
- 유출(F_t<0): 과체중 자산(I^-={d_i<0})에 비례 매도 Δ_i^sell=(|d_i|/S^-)·min(|F_t|, S^-), S^-=Σ_{I^-}|d_i|
  (S^-=0이면 Δ_i^sell = w_i·|F_t|)
"""

from __future__ import annotations

# ── 표준 라이브러리 우선
from typing import Any

# ── 서드파티
import pandas as pd

__all__ = ["build_rebalancing_spec_ts", "build_rebalancing_spec_t"]


# --------------------------- 유틸/검증 ---------------------------

def _ensure_weights_df(
    target_weights: pd.Series | pd.DataFrame,
    index: pd.Index,
    columns: pd.Index,
    *,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """w를 values(index, columns)에 정렬하고 행합=1, 음수 없음 검증."""
    if isinstance(target_weights, pd.Series):
        w_row = target_weights.reindex(columns).fillna(0.0).astype(float)
        w = pd.DataFrame([w_row.values] * len(index), index=index, columns=columns)
    elif isinstance(target_weights, pd.DataFrame):
        w = target_weights.reindex(index=index, columns=columns).fillna(0.0).astype(float)
    else:
        raise TypeError("target_weights는 pandas Series 또는 DataFrame 이어야 합니다.")

    if (w < 0).any().any():
        raise ValueError("target_weights에 음수 항목이 있습니다.")
    row_sum = w.sum(axis=1)
    bad = (row_sum - 1.0).abs() > tol
    if bad.any():
        pos = bad.idxmax()
        raise ValueError(f"target_weights의 행 합이 1이 아닙니다 (예: {pos}, 합={row_sum.loc[pos]:.6f})")
    return w


def _ensure_cashflow_series(cash_flow: float | int | pd.Series, index: pd.Index) -> pd.Series:
    """cash_flow를 index에 정렬된 float Series로 변환."""
    if isinstance(cash_flow, pd.Series):
        return cash_flow.reindex(index).fillna(0.0).astype(float)
    return pd.Series(float(cash_flow), index=index, dtype=float)


# --------------------------- 메인 빌더(TS) ---------------------------

def build_rebalancing_spec_ts(
    values: pd.DataFrame,
    target_weights: pd.Series | pd.DataFrame,
    cash_flow: float | int | pd.Series,
) -> dict[str, Any]:
    """
    시계열 입력에서 리밸런싱 스펙 생성.
    반환: {
      "target_weights": DataFrame(index=values.index, columns=values.columns),
      "cash_flow":     Series(index=values.index),
      "buy_notional":  DataFrame(index=values.index, columns=values.columns),
      "sell_notional": DataFrame(index=values.index, columns=values.columns),
    }
    """
    if not isinstance(values, pd.DataFrame):
        raise TypeError("values는 DataFrame이어야 합니다 (행: 시점, 열: 자산).")

    if values.empty:
        idx, cols = values.index, values.columns
        empty_df = pd.DataFrame(0.0, index=idx, columns=cols)
        empty_sr = pd.Series(0.0, index=idx, dtype=float)
        return {
            "target_weights": empty_df.copy(),
            "cash_flow": empty_sr.copy(),
            "buy_notional": empty_df.copy(),
            "sell_notional": empty_df.copy(),
        }

    # 정렬 및 타입 보정
    values = values.astype(float).sort_index()
    w = _ensure_weights_df(target_weights, index=values.index, columns=values.columns)
    F = _ensure_cashflow_series(cash_flow, index=values.index)

    # 핵심 변수
    P_prev = values.sum(axis=1)            # P_{t-1}
    P_star = P_prev + F                    # P*
    T = w.mul(P_star, axis=0)              # T_i
    d = T - values                         # d_i
    d_pos = d.clip(lower=0.0)              # 미달(+)
    d_neg_mag = (-d.clip(upper=0.0))       # 과체중의 |d|

    S_plus = d_pos.sum(axis=1)             # S^+
    S_minus = d_neg_mag.sum(axis=1)        # S^-

    # 유입(F>=0): 미달 자산에만 비례 매수
    F_in = F.clip(lower=0.0)
    scale_buy = (
        F_in.where(S_plus > 0.0, 0.0)
        .clip(upper=S_plus)
        .div(S_plus.replace(0.0, pd.NA))
        .fillna(0.0)
    )
    buy = d_pos.mul(scale_buy, axis=0).clip(lower=0.0)

    # 유출(F<0): 과체중 자산에서만 비례 매도
    F_out_mag = (-F.clip(upper=0.0))  # |F_t|
    scale_sell = (
        F_out_mag.where(S_minus > 0.0, 0.0)
        .clip(upper=S_minus)
        .div(S_minus.replace(0.0, pd.NA))
        .fillna(0.0)
    )
    sell = d_neg_mag.mul(scale_sell, axis=0)

    # 과체중 없음(S^-=0)인데 유출이 있는 시점 → w_i 비례로 매도
    mask_fallback = (S_minus <= 0.0) & (F_out_mag > 0.0)
    if mask_fallback.any():
        fallback_amt = F_out_mag[mask_fallback]
        sell.loc[mask_fallback] = w.loc[mask_fallback].mul(fallback_amt, axis=0)

    sell = sell.clip(lower=0.0)

    return {
        "target_weights": w,
        "cash_flow": F,
        "buy_notional": buy,
        "sell_notional": sell,
    }


# --------------------------- 단일 시점 빌더 ---------------------------

def build_rebalancing_spec_t(
    values_t: pd.Series,
    target_weights: pd.Series,
    cash_flow_t: float | int,
    *,
    at: Any | None = None,
) -> dict[str, Any]:
    """
    단일 시점 입력에서 리밸런싱 스펙 생성.
    - at: 단일 레코드의 index 라벨(없으면 0 사용)
    """
    if not isinstance(values_t, pd.Series):
        raise TypeError("values_t는 Series여야 합니다 (index: 자산).")
    if not isinstance(target_weights, pd.Series):
        raise TypeError("target_weights는 Series여야 합니다 (index: 자산).")

    idx_label = 0 if at is None else at
    values_df = pd.DataFrame([values_t.astype(float)], index=[idx_label])
    weights_sr = target_weights.astype(float)

    spec = build_rebalancing_spec_ts(values_df, weights_sr, float(cash_flow_t))

    # 단일 시점 접근성을 위해 Series도 함께 제공
    return {
        "target_weights": spec["target_weights"].iloc[0],
        "cash_flow": float(spec["cash_flow"].iloc[0]),
        "buy_notional": spec["buy_notional"].iloc[0],
        "sell_notional": spec["sell_notional"].iloc[0],
        # 원본 DataFrame/Series 형태도 유지
        "target_weights_df": spec["target_weights"],
        "cash_flow_series": spec["cash_flow"],
        "buy_notional_df": spec["buy_notional"],
        "sell_notional_df": spec["sell_notional"],
    }

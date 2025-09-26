"""
Fixed Fractional 사이징 메타 엔진: 계좌·상품 메타데이터를 검증하고 f% 위험 규칙으로 자산 유형별 수량(Q)을 산출, 로트/호가 단위로 하향 보정하며 포트폴리오 위험 한도(h_max)를 강제합니다.

이 파일의 목적:
- 통화/단위 일관성을 전제로 위험 예산 R=f·E와 스탑 거리 기반 분모를 계산해 자산 유형(주식/선물/FX)별 이론 수량 Q를 산출합니다.
- 실행 가능 수량(Q_exec)을 lot_step에 맞춰 내림 보정하고, 다중 주문의 누적 위험이 h_max를 넘지 않도록 컷오프/스케일 방식을 제공합니다.

사용되는 변수와 함수 목록:
- 변수
  - 없음

- 함수
  - _round_down_step(q: float, step: float)
    - 역할: 주어진 step 배수로 수량을 하향 보정
    - 입력값: q: float — 원시 수량, step: float>0 — 로트/호가 단위
    - 반환값: float — step 배수로 내림한 수량

  - risk_budget(E: float, f: float)
    - 역할: 트레이드당 위험 예산 R=f·E 계산
    - 입력값: E: float>0 — 계좌 자본, f: float>0 — 위험 비율(예: 0.01)
    - 반환값: float — 위험 예산 R

  - _size_price_asset(E: float, f: float, entry: float, stop: float, lot_step: float, max_qty: float|None)
    - 역할: 가격자산(주식/코인) 수량 계산(Q, Q_exec, r_exec)
    - 입력값: entry>0, stop≥0, lot_step>0, max_qty≥0|None
    - 반환값: dict — {"R","D","Q","Q_exec","r_exec"}

  - _size_futures(E: float, f: float, entry: float, stop: float, point_value: float, lot_step: float, max_qty: float|None)
    - 역할: 선물 수량 계산(Q, Q_exec, r_exec) — 분모 D·V
    - 입력값: point_value>0, 기타 동일
    - 반환값: dict — {"R","D","Q","Q_exec","r_exec"}

  - _size_fx(E: float, f: float, entry: float, stop: float, pip_value: float, pip_size: float, lot_step: float, max_qty: float|None)
    - 역할: FX 수량 계산(Q, Q_exec, r_exec) — 분모 D_pips·PV
    - 입력값: pip_value>0, pip_size>0, 기타 동일
    - 반환값: dict — {"R","D_pips","Q","Q_exec","r_exec"}

  - size_fixed_fractional_meta(E: float, f: float, entry: float, stop: float, *, asset_type: str="price", lot_step: float=1.0, point_value: float|None=None, pip_value: float|None=None, pip_size: float|None=None, max_qty: float|None=None)
    - 역할: 자산 유형(price/futures/fx)에 따라 해당 사이징 경로를 호출
    - 입력값: 필수 E,f,entry,stop; 유형별 point_value 또는 pip_value/pip_size; lot_step>0; max_qty(선택)
    - 반환값: dict — 자산별 사이징 결과(키는 위 각 경로와 동일)

  - update_equity(E: float, pnl: float)
    - 역할: 트레이드 종료 후 자본 업데이트
    - 입력값: E — 기존 자본, pnl — 실현 손익
    - 반환값: float — 갱신된 자본

  - enforce_portfolio_heat(orders: list[dict], h_max: float, mode: str="cutoff")
    - 역할: 다중 주문의 누적 위험(r_exec)의 상한 h_max 강제(cutoff 또는 scale)
    - 입력값: orders — 각 주문 dict(최소 "Q_exec","r_exec","lot_step" 포함), h_max>0, mode∈{"cutoff","scale"}
    - 반환값: list[dict] — h_max를 만족하도록 조정된 주문 리스트

파일의 흐름(→ / ->):
- 입력 검증(E,f, lot_step 및 유형별 단위) → 위험 예산 R=f·E 계산 → 자산 유형별 분모 산출(D, D·V, D_pips·PV)과 이론 수량 Q
  -> lot_step 하향 보정으로 Q_exec 산출(+max_qty 캡) → r_exec 계산(E 기준 위험 비율)
  -> 여러 주문에 대해 enforce_portfolio_heat로 누적 위험이 h_max 이내가 되도록 컷오프/스케일 적용
"""

from decimal import Decimal
from typing import Optional, Dict, Any, List


def _round_down_step(q: float, step: float) -> float:
    if step <= 0:
        raise ValueError("step must be > 0")
    dq = Decimal(str(q))
    ds = Decimal(str(step))
    return float((dq // ds) * ds)


def risk_budget(E: float, f: float) -> float:
    if E <= 0 or f <= 0:
        raise ValueError("E and f must be > 0")
    return f * E


def _size_price_asset(E: float, f: float, entry: float, stop: float, lot_step: float, max_qty: Optional[float]):
    if entry <= 0 or stop < 0:
        raise ValueError("entry must be > 0 and stop >= 0")
    D = entry - stop
    if D <= 0:
        return {"R": risk_budget(E, f), "D": D, "Q": 0.0, "Q_exec": 0.0, "r_exec": 0.0}
    R = risk_budget(E, f)
    Q_raw = R / D
    if max_qty is not None:
        Q_raw = min(Q_raw, max_qty)
    Q_exec = _round_down_step(Q_raw, lot_step)
    r_exec = (Q_exec * D) / E if E > 0 else 0.0
    return {"R": R, "D": D, "Q": Q_raw, "Q_exec": Q_exec, "r_exec": r_exec}


def _size_futures(E: float, f: float, entry: float, stop: float, point_value: float, lot_step: float, max_qty: Optional[float]):
    if entry <= 0 or stop < 0 or point_value <= 0:
        raise ValueError("entry, point_value must be > 0 and stop >= 0")
    D = entry - stop
    if D <= 0:
        return {"R": risk_budget(E, f), "D": D, "Q": 0.0, "Q_exec": 0.0, "r_exec": 0.0}
    R = risk_budget(E, f)
    denom = D * point_value
    Q_raw = R / denom
    if max_qty is not None:
        Q_raw = min(Q_raw, max_qty)
    Q_exec = _round_down_step(Q_raw, lot_step)
    r_exec = (Q_exec * denom) / E if E > 0 else 0.0
    return {"R": R, "D": D, "Q": Q_raw, "Q_exec": Q_exec, "r_exec": r_exec}


def _size_fx(E: float, f: float, D_pips: float, pip_value: float, lot_step: float, max_qty: Optional[float]) -> Dict[str, Any]:
    if E <= 0 or f <= 0 or D_pips <= 0 or pip_value <= 0:
        raise ValueError("E,f,D_pips,pip_value must be > 0")
    R = risk_budget(E, f)
    denom = D_pips * pip_value
    Q_raw = R / denom
    if max_qty is not None:
        Q_raw = min(Q_raw, max_qty)
    Q_exec = _round_down_step(Q_raw, lot_step)
    r_exec = (Q_exec * denom) / E if E > 0 else 0.0
    return {"R": R, "D_pips": D_pips, "Q": Q_raw, "Q_exec": Q_exec, "r_exec": r_exec}


def size_fixed_fractional_meta(
    E: float,
    f: float,
    entry: float,
    stop: float,
    *,
    asset_type: str = "price",
    lot_step: float = 1.0,
    point_value: Optional[float] = None,
    pip_value: Optional[float] = None,
    pip_size: Optional[float] = None,
    max_qty: Optional[float] = None,
) -> Dict[str, Any]:
    if E <= 0 or f <= 0:
        raise ValueError("E and f must be > 0")
    if lot_step <= 0:
        raise ValueError("lot_step must be > 0")
    t = asset_type.lower()
    if t == "price":
        return _size_price_asset(E, f, entry, stop, lot_step, max_qty)
    if t == "futures":
        if point_value is None:
            raise ValueError("point_value required for futures")
        return _size_futures(E, f, entry, stop, point_value, lot_step, max_qty)
    if t == "fx":
        if pip_value is None or pip_size is None:
            raise ValueError("pip_value and pip_size required for fx")
        return _size_fx(E, f, entry, stop, pip_value, pip_size, lot_step, max_qty)
    raise ValueError("asset_type must be one of {'price','futures','fx'}")


def update_equity(E: float, pnl: float) -> float:
    return E + pnl


def enforce_portfolio_heat(
    orders: List[Dict[str, Any]],
    h_max: float,
    mode: str = "cutoff",
) -> List[Dict[str, Any]]:
    if h_max <= 0:
        raise ValueError("h_max must be > 0")
    if not orders:
        return []
    enriched = []
    for i, o in enumerate(orders):
        r = float(o.get("r_exec", 0.0))
        s = float(o.get("lot_step", 1.0))
        q = float(o.get("Q_exec", 0.0))
        p = int(o.get("priority", i))
        enriched.append({"idx": i, "priority": p, "r_exec": r, "Q_exec": q, "lot_step": s, "obj": o})
    if mode == "cutoff":
        enriched.sort(key=lambda x: (x["priority"], x["idx"]))
        out = []
        acc = 0.0
        for e in enriched:
            take = acc + e["r_exec"] <= h_max
            if not take:
                e["obj"]["Q_exec"] = 0.0
                e["obj"]["r_exec"] = 0.0
            else:
                acc += e["r_exec"]
            out.append(e["obj"])
        return out
    if mode == "scale":
        total = sum(e["r_exec"] for e in enriched)
        if total <= h_max or total == 0:
            return [e["obj"] for e in enriched]
        scale = h_max / total
        out = []
        for e in enriched:
            q_scaled = _round_down_step(e["Q_exec"] * scale, e["lot_step"])
            if q_scaled <= 0:
                e["obj"]["Q_exec"] = 0.0
                e["obj"]["r_exec"] = 0.0
            else:
                if e["Q_exec"] > 0:
                    e["obj"]["r_exec"] = e["r_exec"] * (q_scaled / e["Q_exec"])
                else:
                    e["obj"]["r_exec"] = 0.0
                e["obj"]["Q_exec"] = q_scaled
            out.append(e["obj"])
        return out
    raise ValueError("mode must be 'cutoff' or 'scale'")

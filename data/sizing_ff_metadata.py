from decimal import Decimal
from typing import Any, Dict, List, Optional


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


def _size_price_asset(
    E: float,
    f: float,
    entry: float,
    stop: float,
    lot_step: float,
    max_qty: Optional[float],
) -> Dict[str, Any]:
    if entry <= 0 or stop < 0:
        raise ValueError("entry>0 and stop>=0 required")
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


def _size_futures(
    E: float,
    f: float,
    entry: float,
    stop: float,
    point_value: float,
    lot_step: float,
    max_qty: Optional[float],
) -> Dict[str, Any]:
    if entry <= 0 or stop < 0 or point_value <= 0:
        raise ValueError("entry>0, stop>=0, point_value>0 required")
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


def _size_fx(
    E: float,
    f: float,
    D_pips: float,
    pip_value: float,
    lot_step: float,
    max_qty: Optional[float],
) -> Dict[str, Any]:
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
        if entry <= 0 or stop < 0 or pip_size <= 0:
            raise ValueError("entry>0, stop>=0, pip_size>0 required for fx")
        D_pips = (entry - stop) / pip_size
        if D_pips <= 0:
            R = risk_budget(E, f)
            return {"R": R, "D_pips": D_pips, "Q": 0.0, "Q_exec": 0.0, "r_exec": 0.0}
        return _size_fx(E, f, D_pips, pip_value, lot_step, max_qty)

    raise ValueError("asset_type must be one of {'price', 'futures', 'fx'}")


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

    enriched: List[Dict[str, Any]] = []
    for i, o in enumerate(orders):
        r = float(o.get("r_exec", 0.0))
        s = float(o.get("lot_step", 1.0))
        q = float(o.get("Q_exec", 0.0))
        p = int(o.get("priority", i))
        enriched.append(
            {"idx": i, "priority": p, "r_exec": r, "Q_exec": q, "lot_step": s, "obj": o}
        )

    if mode == "cutoff":
        enriched.sort(key=lambda x: (x["priority"], x["idx"]))
        out: List[Dict[str, Any]] = []
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
        out: List[Dict[str, Any]] = []
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

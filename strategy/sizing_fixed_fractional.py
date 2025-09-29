import math

def risk_budget(E: float, f: float) -> float:
    if E <= 0 or f <= 0:
        raise ValueError("E>0 and f>0 required")
    return f * E

def stop_distance_from_prev_low(entry: float, prev_low: float) -> float:
    if not (entry > 0 and prev_low >= 0):
        return float("nan")
    return float(entry) - float(prev_low)

def floor_to_step(q: float, step: float) -> float:
    if step <= 0:
        raise ValueError("step>0 required")
    return math.floor(float(q) / step) * step

def qty_stock_coin_ff(E: float, f: float, entry: float, prev_low: float, *, lot_step: float = 1.0) -> float:
    if E <= 0 or f <= 0 or entry <= 0 or prev_low < 0 or lot_step <= 0:
        return 0.0
    D = stop_distance_from_prev_low(entry, prev_low)
    if D <= 0:
        return 0.0
    Q_raw = (f * E) / D
    return floor_to_step(Q_raw, lot_step)

def qty_futures_ff(E: float, f: float, entry: float, prev_low: float, V: float, *, lot_step: float = 1.0) -> float:
    if E <= 0 or f <= 0 or entry <= 0 or prev_low < 0 or V <= 0 or lot_step <= 0:
        return 0.0
    D = stop_distance_from_prev_low(entry, prev_low)
    if D <= 0:
        return 0.0
    Q_raw = (f * E) / (D * V)
    return floor_to_step(Q_raw, lot_step)

def qty_fx_ff(E: float, f: float, D_pips: float, PV: float, *, lot_step: float = 1.0) -> float:
    if E <= 0 or f <= 0 or D_pips <= 0 or PV <= 0 or lot_step <= 0:
        return 0.0
    Q_raw = (f * E) / (float(D_pips) * float(PV))
    return floor_to_step(Q_raw, lot_step)

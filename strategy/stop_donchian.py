from collections import deque

def donchian_prev_low(lows, n):
    if n <= 0:
        raise ValueError("n must be positive")
    m = len(lows)
    out = [None] * m
    dq = deque()
    for i in range(m):
        v = lows[i]
        while dq and dq[-1][1] >= v:
            dq.pop()
        dq.append((i, v))
        while dq and dq[0][0] <= i - n:
            dq.popleft()
        t = i + 1
        if t < m and i >= n - 1:
            out[t] = dq[0][1]
    return out

def donchian_stop_hits(lows, n):
    prev = donchian_prev_low(lows, n)
    m = len(lows)
    hits = [False] * m
    for t in range(m):
        p = prev[t]
        if p is not None and lows[t] <= p:
            hits[t] = True
    return hits

class DonchianStopLong:
    __slots__ = ("n", "_dq", "_count")

    def __init__(self, n):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n
        self._dq = deque()
        self._count = 0

    @property
    def prev_low(self):
        if self._count >= self.n and self._dq:
            return self._dq[0][1]
        return None

    def on_bar_open(self):
        return self.prev_low

    def on_bar_close(self, low):
        hit = False
        pl = self.prev_low
        if pl is not None and low <= pl:
            hit = True
        i = self._count
        while self._dq and self._dq[-1][1] >= low:
            self._dq.pop()
        self._dq.append((i, low))
        while self._dq and self._dq[0][0] <= i - self.n:
            self._dq.popleft()
        self._count += 1
        return hit

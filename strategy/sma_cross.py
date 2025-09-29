from collections import deque

class RollingSMA:
    def __init__(self, n):
        if n <= 0:
            raise ValueError("window n must be positive")
        self.n = n
        self._buf = deque(maxlen=n)
        self._sum = 0.0
        self._value = None

    @property
    def ready(self):
        return len(self._buf) == self.n

    @property
    def value(self):
        return self._value if self.ready else None

    def update(self, x):
        if len(self._buf) == self._buf.maxlen:
            oldest = self._buf[0]
            self._sum += x - oldest
            self._buf.append(x)
            self._value = self._sum / self.n
        else:
            self._buf.append(x)
            self._sum += x
            self._value = self._sum / self.n if self.ready else None
        return self._value


class SmaCrossLongOnlyStateEngine:
    __slots__ = ("short", "long", "epsilon", "_pending_position", "position")

    def __init__(self, short_n=10, long_n=50, epsilon=0.0):
        if short_n <= 0 or long_n <= 0:
            raise ValueError("window sizes must be positive")
        if short_n >= long_n:
            raise ValueError("short_n must be smaller than long_n")
        self.short = RollingSMA(int(short_n))
        self.long = RollingSMA(int(long_n))
        self.epsilon = max(0.0, float(epsilon))
        self._pending_position = 0
        self.position = 0

    def on_bar_close(self, close):
        self.short.update(close)
        self.long.update(close)
        if self.short.ready and self.long.ready:
            diff = self.short.value - self.long.value
            self._pending_position = 1 if diff > self.epsilon else 0
        else:
            self._pending_position = 0

    def on_bar_open(self):
        self.position = self._pending_position
        return self.position


def sma_positions(close_seq, epsilon=0.0, short_n=10, long_n=50):
    short = RollingSMA(int(short_n))
    long_  = RollingSMA(int(long_n))
    positions = []
    eps = max(0.0, float(epsilon))
    for px in close_seq:
        s_prev = short.value
        l_prev = long_.value
        positions.append(1 if (s_prev is not None and l_prev is not None and (s_prev - l_prev) > eps) else 0)
        short.update(px)
        long_.update(px)
    return positions

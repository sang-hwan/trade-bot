"""
이 파일의 목적:
- 증분 SMA와 룩어헤드 금지 규칙을 이용해 SMA 단기/장기 크로스(롱 온리) 시그널을 스트리밍/배치 방식으로 산출합니다.

사용되는 변수와 함수 목록:
- 변수
  - epsilon: float — 동률/미세 진동 완화 임계값(기본 0.0, 0 이상 권장)
- 클래스/함수
  - RollingSMA(n): O(1) 증분 갱신식 기반 고정창 단순이동평균
    - 입력값: n(int>0) — 윈도 길이
    - 출력값: update(x: float) -> float|None — 충분 샘플 이전 None, 이후 SMA 값
  - SmaCrossLongOnlyStateEngine(short_n=10, long_n=50, epsilon=0.0): 스트리밍 시그널 엔진
    - 입력값: short_n(int>0), long_n(int>0, short_n<long_n), epsilon(float>=0 권장)
    - 출력값: on_bar_open() -> int — 현재 봉 오픈에서 적용될 상태(1=롱, 0=관망)
             on_bar_close(close: float) -> None — 종가로 SMA 갱신 및 다음 봉 상태 확정
  - sma_positions(close_seq, epsilon=0.0, short_n=10, long_n=50) -> list[int]
    - 입력값: close_seq(iterable of float), epsilon/short_n/long_n
    - 출력값: 각 시점의 포지션(0/1) 리스트 — t 오픈에서 적용될 상태

파일의 흐름(→):
- (스트리밍) on_bar_open() → on_bar_close(close) … 반복
- (배치) sma_positions(): 각 t에서 t−1의 SMA 비교로 포지션 결정 후 SMA 갱신
"""

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
        self.short = RollingSMA(short_n)
        self.long = RollingSMA(long_n)
        self.epsilon = float(epsilon)
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
    short = RollingSMA(short_n)
    long_ = RollingSMA(long_n)
    positions = []
    eps = float(epsilon)
    for px in close_seq:
        s_prev = short.value
        l_prev = long_.value
        positions.append(1 if (s_prev is not None and l_prev is not None and (s_prev - l_prev) > eps) else 0)
        short.update(px)
        long_.update(px)
    return positions

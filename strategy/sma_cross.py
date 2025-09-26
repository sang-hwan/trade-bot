"""
이 파일의 목적:
- 증분 SMA와 룩어헤드 금지 규칙으로 SMA 단기/장기 크로스(롱 온리) 시그널을 스트리밍/배치로 산출합니다.

사용되는 변수와 함수 목록:
- 변수
  - position: int(0/1) — 현재 봉 오픈에서 적용될 상태(엔진 인스턴스의 상태 변수)
  - _pending_position: int(0/1) — 다음 봉 오픈에 적용될 예정 상태(엔진 인스턴스의 상태 변수)
  - epsilon: float(≥0 권장) — 동률/미세 진동 방지 임계값

- 함수
  - RollingSMA(n): 고정창 단순이동평균을 O(1)로 증분 갱신
    - 입력값: n=필수(int>0) — 윈도 길이
    - 출력값: update(x: float) -> float|None — SMA 값(윈도 충족 전에는 None)
  - RollingSMA.ready() [property]: 윈도 충족 여부
    - 입력값: (없음)
    - 출력값: bool — 샘플이 n개 채워졌는지
  - RollingSMA.value() [property]: 최신 SMA 값
    - 입력값: (없음)
    - 출력값: float|None — ready 이전 None, 이후 SMA 값
  - SmaCrossLongOnlyStateEngine(short_n=10, long_n=50, epsilon=0.0): 스트리밍 시그널 엔진
    - 입력값: short_n(int>0), long_n(int>0, short_n<long_n), epsilon(float≥0 권장)
    - 출력값: on_bar_open() -> int — 현재 봉 오픈에 적용할 상태(1=롱, 0=관망)
             on_bar_close(close: float) -> None — 종가로 두 SMA 갱신 후 다음 봉 상태 확정
  - sma_positions(close_seq, epsilon=0.0, short_n=10, long_n=50): 배치 방식 포지션 산출
    - 입력값: close_seq(iterable[float]), epsilon(float≥0), short_n(int>0), long_n(int>0)
    - 출력값: list[int] — 각 t에서의 포지션(0/1), 길이는 close_seq와 동일

파일의 흐름(→):
- (스트리밍) on_bar_open() → on_bar_close(close) → … 반복
- (배치) 각 t에서 전일 SMA 비교로 포지션 결정 → 같은 t의 종가로 SMA 갱신
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

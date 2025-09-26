"""
증분 SMA로 룩어헤드 없이 SMA 단기/장기 크로스(롱 온리) 포지션을 스트리밍·배치로 산출

이 파일의 목적:
- 고정창 단순이동평균(SMA)을 O(1)로 갱신하여 계산 비용을 낮추고, 전일 확정치로만 의사결정해 룩어헤드를 방지합니다.
- 스트리밍 엔진은 “전 봉 마감에서 신호 확정 → 다음 봉 오픈에서 실행” 규칙을 일관 적용합니다.
- 배치 함수는 스트리밍과 동일한 실행 의미론을 모사하여 재현 가능한 결과를 제공합니다.

사용되는 변수와 함수 목록:
- 변수
  - short: RollingSMA — 단기 SMA 집계기(윈도 길이 = short_n)
  - long: RollingSMA — 장기 SMA 집계기(윈도 길이 = long_n)
  - position: int(0/1) — 현재 봉 오픈에 적용될 상태(실행된 포지션)
  - _pending_position: int(0/1) — 다음 봉 오픈에 적용될 예정 상태(예약 포지션)
  - epsilon: float(권장 ≥ 0) — 동률/미세 진동 완충 임계값(diff ≤ epsilon이면 관망)

- 함수
  - RollingSMA(n: int)
    - 역할: 고정창 단순이동평균을 O(1)로 증분 갱신
    - 입력값: n: int (> 0) - 윈도 길이
    - 반환값: update(x: float) -> float | None - 최신 SMA(윈도 충족 전에는 None)
  - RollingSMA.ready [property]
    - 역할: 샘플 수가 n에 도달했는지 여부
    - 입력값: (없음)
    - 반환값: bool - 윈도 충족 여부
  - RollingSMA.value [property]
    - 역할: 최신 SMA 값 조회
    - 입력값: (없음)
    - 반환값: float | None - ready 이전에는 None
  - SmaCrossLongOnlyStateEngine(short_n: int = 10, long_n: int = 50, epsilon: float = 0.0)
    - 역할: 전일 SMA 크로스(diff = SMA_short - SMA_long)를 기준으로 다음 봉 오픈 포지션(롱/관망) 결정
    - 입력값: short_n: int (> 0), long_n: int (> 0, short_n < long_n), epsilon: float(권장 ≥ 0) - 크로스 임계
    - 반환값: on_bar_open() -> int(0/1), on_bar_close(close: float) -> None
  - SmaCrossLongOnlyStateEngine.on_bar_close(close: float)
    - 역할: 종가로 두 SMA를 갱신하고 diff를 계산하여 다음 봉 예약 포지션(_pending_position) 결정
    - 입력값: close: float - 해당 봉 종가(단일 타임존/정렬 가정)
    - 반환값: None
  - SmaCrossLongOnlyStateEngine.on_bar_open()
    - 역할: 예약된 _pending_position을 현재 position으로 확정(실행)
    - 입력값: (없음)
    - 반환값: int(0/1) - 현재 봉 오픈 적용 포지션
  - sma_positions(close_seq: iterable[float], epsilon: float = 0.0, short_n: int = 10, long_n: int = 50)
    - 역할: 배치 모드로 포지션 열 벡터를 생성(각 t에서 전일 SMA 비교로 결정 후 같은 t 종가로 SMA 갱신)
    - 입력값: close_seq: iterable[float] - 종가 시계열(연속 관측 ≥ long_n 권장)
              epsilon: float(권장 ≥ 0)
              short_n: int (> 0)
              long_n: int (> 0, short_n < long_n)
    - 반환값: list[int] - 각 시점 포지션(0/1), 길이 = close_seq

파일의 흐름(→ / ->):
- 스트리밍: on_bar_open() -> on_bar_close(close) -> … 반복(두 SMA가 ready 전까지는 항상 관망=0)
- 배치: 각 t에서 (SMA_short_{t-1} - SMA_long_{t-1}) > epsilon ? 1 : 0 -> 같은 t 종가로 두 SMA 갱신
- 동률/진동 완충: diff ≤ epsilon이면 관망(0) 처리로 스위칭 과민 반응을 억제
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

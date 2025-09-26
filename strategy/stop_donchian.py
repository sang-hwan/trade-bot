"""
N-일 Donchian 최저값 기반 롱 포지션 스탑 트리거를 룩어헤드 없이 배치·스트리밍으로 산출합니다.

이 파일의 목적:
- 전일 기준 임계값 L_N(t-1)을 모노토닉 덱으로 O(m) 시간에 생성하고,
  현재 저가가 L_N(t-1) 이하인지(스탑 트리거) 판정합니다.
- 동일 로직을 배치(process all bars)와 스트리밍(on_bar_open/close) 방식 모두로 제공해
  백테스트·실거래 엔진에 일관되게 적용합니다.

사용되는 변수와 함수 목록:
- 변수
  - 없음

- 함수
  - donchian_prev_low(lows, n)
    - 역할: 전일 기준 N-일 최저 임계값 L_N(t-1) 시퀀스를 생성
    - 입력값: lows: 시퀀스[float] - 각 시점의 저가, n: int(>0) - 윈도 길이
    - 반환값: list[float | None] - 각 t에서의 L_N(t-1); 초기 n 구간은 None

  - donchian_stop_hits(lows, n)
    - 역할: 스탑 트리거(Boolean) 시퀀스 계산: low_t ≤ L_N(t-1)
    - 입력값: lows: 시퀀스[float], n: int(>0)
    - 반환값: list[bool] - 각 t에서 트리거 여부(True/False)

  - DonchianStopLong(n)
    - 역할: 롱 기준 스트리밍 스탑 엔진(룩어헤드 금지)
    - 입력값: n: int(>0)
    - 반환값:
        - prev_low [property] -> float | None - 현재 시점에서 '다음 봉 오픈에 적용될' L_N(t-1)
        - on_bar_open() -> float | None - 현재 봉 오픈에서 적용할 L_N(t-1) 반환
        - on_bar_close(low: float) -> bool - 저가로 내부 상태 갱신 및 (low ≤ L_N(t-1)) 판정

파일의 흐름(→ / ->):
- 스트리밍: on_bar_open()로 L_N(t-1) 확인 -> on_bar_close(low)로 덱 갱신 및 트리거 판정 -> 반복
- 배치: donchian_prev_low(lows, n)로 L_N(t-1) 시퀀스 생성 -> donchian_stop_hits(lows, n)로 트리거 시퀀스 계산
"""

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

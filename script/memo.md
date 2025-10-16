- 모니터링 UX 개선
  - Trade Log 표시 안 나오는 문제 해결
  > Trade Log Viewer에 trades.csv fallback 로직 추가 또는 산출물에 trades.jsonl 추가 기록
  - 윈도우 로그 보존
  > PowerShell: 실매매 프로세스 시작 시 표준출력/에러를 live_bot.log로 리다이렉트.
  > 이러면 PID 사라질 때도 로그로 사유 파악 용이
  - plan/ 폴더 정책 결정
  > 단순화 원함 → 스크립트 --plan-out $LIVE_RUN_DIR로 바꾸어 폴더 제거
  - 세션 탐색 통일
  > 공용 세션 탐색 유틸 도입, 모든 뷰어에서 동일 기준 사용.

- 대기(Idle) & 재시도 추가
> 스케쥴러로 대체

- 전략 전환(대형주/소형주, 우상향/우하향/박스권)
> 미래 예측 불가로 인한 우상향/우하향/박스권 예측 불가
> 전략 조사

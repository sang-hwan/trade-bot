```powershell
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
.\.venv\Scripts\Activate.ps1

# 비활성화
deactivate

# 필요 라이브러리 설치
pip install -r requirements.txt

# 데이터 준비 및 시그널 생성 스크립트 실행
# 예시: SPY 티커의 2020-01-01부터 현재까지 데이터를 받아 ./data 폴더에 Parquet 파일로 저장
python .\data\prep_sma_signal.py --ticker "SPY" --start "2020-01-01" --out ".\data"

# 백테스터 실행
python .\simulation\simple_backtester.py
```
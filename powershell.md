```powershell
# 흔한 불필요 폴더 제외하고 찾기(.git, node_modules, venv 등)
$rx = '\\(\.git|node_modules|venv|\.venv|dist|build|__pycache__)(\\|$)'  # 제외 경로 정규식

Get-ChildItem -Recurse -File -Filter 'features_sma.py' |
  Where-Object { $_.FullName -notmatch $rx } |
  ForEach-Object {
    [PSCustomObject]@{
      Path     = (Resolve-Path $_.FullName -Relative)
      Size     = $_.Length
      Modified = $_.LastWriteTime
    }
  } | Format-Table -Auto

# 가상환경 설치
python -m venv .venv

# 가상환경 실행
.\.venv\Scripts\Activate.ps1

# config.py 실행
#   --tickers        대상 심볼(공백 구분 다중)
#   --start          시작일(UTC, YYYY-MM-DD)
#   --end            종료일('today' 허용 → 실행 시점 UTC로 동결)
#   --timeframe      캔들 주기(1D/1H/15m/5m)
#   --fee-bps        수수료 가정(bps, 5=0.05%)
#   --slippage-bps   슬리피지 가정(bps, 10=0.10%)
#   --out            설정 저장 폴더
python ".\data\config.py" `
  --tickers AAPL MSFT `
  --start 2020-01-01 `
  --end today `
  --timeframe 1D `
  --fee-bps 5 `
  --slippage-bps 10 `
  --out ".\data\config_snapshots"

# load_ohlcv.py 실행
#   --config  config.json 경로
#   --out     OHLCV 저장 루트(하위에 주기별 폴더/파일 생성)
python ".\data\load_ohlcv.py" `
  --config .\data\config_snapshots\config.json `
  --out .\data\ohlcv

# clean_ohlcv.py 실행
#   --config  입력 파일 추론용 config.json 경로
python ".\data\clean_ohlcv.py" `
  --config .\data\config_snapshots\config.json

# features_sma.py 실행
#   --in         입력 파일(.parquet | .csv) — clean 산출물 경로
#                예) .\data\ohlcv\1D\ohlcv_1D_27ba50c1538c_clean.parquet
#   --out        출력 파일 — 입력 형식 유지, '_feat' 접미사 권장
#                (미지정 시 자동으로 동일 규칙: *_clean_feat.<확장자>)
python ".\strategy\features_sma.py" `
  --in  ".\data\ohlcv\1D\ohlcv_1D_27ba50c1538c_clean.parquet" `
  --out ".\data\ohlcv\1D\ohlcv_1D_27ba50c1538c_clean_feat.parquet"
```
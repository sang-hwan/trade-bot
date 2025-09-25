```powershell
# 흔한 불필요 폴더 제외하고 찾기(.git, node_modules, venv 등)
$rx='\\(\.git|node_modules|venv|\.venv|dist|build|__pycache__)(\\|$)';
Get-ChildItem -Recurse -File -Filter "config.py" |
  Where-Object { $_.FullName -notmatch $rx } |
  ForEach-Object {
    [PSCustomObject]@{ Path=(Resolve-Path $_.FullName -Relative); Size=$_.Length; Modified=$_.LastWriteTime }
  } | Format-Table -Auto

# 가상환경 설치
python -m venv .venv

# 가상환경 실행
.\.venv\Scripts\Activate.ps1

# config.py 실행
python ".\data\config.py" --tickers AAPL MSFT --start 2020-01-01 --end today --timeframe 1D --fee-bps 5 --slippage-bps 10 --out ".\data\config_snapshots"

# load_ohlcv.py 실행
python .\data\load_ohlcv.py --config .\data\config_snapshots\config.json --out .\data\ohlcv

# clean_ohlcv.py 실행
python .\data\clean_ohlcv.py --config .\data\config_snapshots\config.json
```
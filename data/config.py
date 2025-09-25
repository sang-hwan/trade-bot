"""
이 파일의 목적
- 백테스트 재현성을 위해, 사용자가 준 상대 입력(예: 'today')을 실행 시점의 **절대값**으로 동결하고, 결과 설정을 사람이 읽기 쉬운 **JSON 파일**로 저장한다.
- 동일 설정으로 재실행 시 동일 결과가 나오도록 **최신 설정(config.json)**과 **시점별 스냅샷(config_YYYYMMDD_HHMMSS.json)** 두 형태를 함께 남긴다.
- 핵심 설정만으로 생성한 **config_id(12자리 해시)**로 서로 다른 실행 간 설정의 동일/상이를 기계적으로 판별할 수 있게 한다.

사용되는 변수와 함수 목록
> 사용되는 변수의 의미
- ALLOWED_TIMEFRAMES: 허용되는 캔들 주기 집합 {"1D","1H","15m","5m"} — 잘못된 주기 입력을 조기에 차단(가드레일).
- args: 명령줄 인자를 담는 argparse.Namespace — main()에서 CLI로 수집된 원시 입력 컨테이너.
- start_abs / end_abs: parse_date로 절대 날짜로 해석한 시작/종료일 — 재현성 확보의 핵심(상대값 제거).
- fee_rate / slippage_rate: bps를 비율로 환산한 값(예: 5bps → 0.0005) — 이후 계산에 바로 사용 가능하도록 전처리.
- core: 해시 생성에 사용되는 **핵심 설정** 딕셔너리(파생값 제외) — 정렬 JSON으로 직렬화 후 SHA-256 해시 입력으로 사용.
- config_id: core를 정렬 JSON으로 해시한 12자리 식별자 — "정말 같은 설정인가?"를 빠르게 판별.
- cfg: 디스크에 기록되는 **최종 설정** 딕셔너리 — core + 파생/메타(resolved_* / *_rate / generated_at_utc / config_id).
- stable / snapshot: 저장된 설정 파일 경로 — stable은 항상 최신, snapshot은 실행 시점별 이력.

> 사용되는 함수의 의미
- parse_date(s): 'today'를 **현재 UTC 날짜**로 치환하고, 그 외에는 'YYYY-MM-DD'를 절대 날짜로 파싱(형식이 틀리면 예외).
- bps_to_rate(bps): 베이시스 포인트(float)를 **비율(float)**로 변환(예: 10.0 → 0.0010).
- build_config(args): 날짜 절대화, 주기 유효성 검사, 비용 환산, 핵심 설정 해시 생성까지 수행해 **최종 설정(cfg)**을 구성.
- save_config(cfg, out_dir): cfg를 **config.json(최신)**과 **config_타임스탬프.json(스냅샷)** 두 파일로 저장.
- main(): CLI 인자를 수집 → cfg 생성/저장 → 요약 정보를 콘솔에 출력(부수효과).

> 파일의 화살표 프로세스
명령줄 옵션 입력 → main에서 인자 파싱 → build_config로 전달
→ (날짜 절대화) → (타임프레임 유효성 검사) → (수수료·슬리피지 환산) → (핵심 설정 core 구성)
→ (core 정렬 JSON 해시 → config_id 생성) → 최종 cfg 생성
→ save_config로 파일 저장(최신·스냅샷) → 콘솔에 요약 출력

> 사용되는 함수의 입력값 목록과 그 의미
- parse_date(s: str): 'YYYY-MM-DD' 또는 'today' — 상대/절대 날짜 문자열.
- bps_to_rate(bps: float): bps 단위의 비용 가정(0 이상 권장) — 수수료/슬리피지 등.
- build_config(args: argparse.Namespace):
  • tickers(list[str]): 대상 심볼 목록(예: ["AAPL","MSFT"])  
  • start(str): 시작일('YYYY-MM-DD' 또는 'today')  
  • end(str): 종료일('YYYY-MM-DD' 또는 'today', 시작일 이상)  
  • timeframe(str): {"1D","1H","15m","5m"} 중 하나  
  • fee_bps(float): 수수료 가정(bps)  
  • slippage_bps(float): 슬리피지 가정(bps)  
  • seed(int): 난수 시드(재현성 관리용)  
  • source(str): 데이터 출처 표기 문자열
- save_config(cfg: dict, out_dir: pathlib.Path): 최종 설정 딕셔너리와 저장 폴더 경로.
- main(): (직접 인자 없음) — CLI에서 수집.

> 사용되는 함수의 출력값 목록과 그 의미
- parse_date → datetime.date: 동결된 **절대 날짜**.
- bps_to_rate → float: **비율** 값(예: 0.0005 = 0.05%).
- build_config → dict(cfg): 저장 가능한 **최종 설정**(파생 필드/메타 포함).
- save_config → (Path, Path): **stable(최신)**, **snapshot(스냅샷)** 파일 경로 튜플.
- main → None: 파일 생성 및 요약 출력(부수효과), 반환값 없음.
"""

import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime, date, timezone

ALLOWED_TIMEFRAMES = {"1D", "1H", "15m", "5m"}

def parse_date(s: str) -> date:
    if s.lower() == "today":
        return datetime.now(timezone.utc).date()
    return date.fromisoformat(s)

def bps_to_rate(bps: float) -> float:
    return float(bps) / 10_000.0

def build_config(args: argparse.Namespace) -> dict:
    start_abs = parse_date(args.start)
    end_abs = parse_date(args.end)
    if end_abs < start_abs:
        raise ValueError(f"end_date({end_abs}) must be >= start_date({start_abs})")
    if args.timeframe not in ALLOWED_TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {sorted(ALLOWED_TIMEFRAMES)}")

    fee_rate = bps_to_rate(args.fee_bps)
    slippage_rate = bps_to_rate(args.slippage_bps)

    core = {
        "tickers": args.tickers,
        "start_date": str(start_abs),
        "end_date": str(end_abs),
        "timeframe": args.timeframe,
        "fee_bps": float(args.fee_bps),
        "slippage_bps": float(args.slippage_bps),
        "seed": int(args.seed),
        "source": args.source,
    }
    config_id = hashlib.sha256(json.dumps(core, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    cfg = {
        **core,
        "resolved_start": str(start_abs),
        "resolved_end": str(end_abs),
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_id": config_id,
    }
    return cfg

def save_config(cfg: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stable = out_dir / "config.json"
    snapshot = out_dir / f"config_{ts}.json"
    text = json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True)
    stable.write_text(text, encoding="utf-8")
    snapshot.write_text(text, encoding="utf-8")
    return stable, snapshot

def main():
    parser = argparse.ArgumentParser(description="초보자용: 백테스트 설정 파일 동결")
    parser.add_argument("--tickers", nargs="+", default=["AAPL"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="today")
    parser.add_argument("--timeframe", default="1D", choices=sorted(ALLOWED_TIMEFRAMES))
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="config_snapshots")
    parser.add_argument("--source", default="yahoo")
    args = parser.parse_args()

    cfg = build_config(args)
    stable, snapshot = save_config(cfg, Path(args.out))

    print("\n[설정 동결 완료]\n")
    print(f"- tickers          : {', '.join(cfg['tickers'])}")
    print(f"- period           : {cfg['resolved_start']} → {cfg['resolved_end']} ({cfg['timeframe']})")
    print(f"- fees/slippage    : {cfg['fee_bps']}bps / {cfg['slippage_bps']}bps"
          f"  (rate={cfg['fee_rate']:.4%}/{cfg['slippage_rate']:.4%})")
    print(f"- seed/source      : {cfg['seed']} / {cfg['source']}")
    print(f"- generated_at_utc : {cfg['generated_at_utc']}")
    print(f"- config_id        : {cfg['config_id']}")
    print(f"- saved            : {stable}  &&  {snapshot}\n")

if __name__ == "__main__":
    main()

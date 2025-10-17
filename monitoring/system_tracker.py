# monitoring/system_tracker.py
"""
시스템 문제 추적 (Streamlit)

공개 API/계약
- 입력: RUNS_ROOT/runs/<session> 하위 상태/로그 파일(관용 경로)
  - 상태(JSON): status/system_status.json, outputs/system_status.json, logs/system_status.json,
                broker/status.json, broker/heartbeat.json, outputs/broker_status.json, logs/broker_status.json
  - 로그(TEXT/JSONL): logs/*.log, logs/*.jsonl (최근 일부만 스캔)
- 출력: 카드·배지 형태로 브로커 연결/인증, 레이트 리밋, 시계 동기화, 거래 캘린더/시가 도달, 이상 로그 개요

타이밍/예외
- 파일은 실시간 append/갱신 가정: JSONL 손상 라인은 json.JSONDecodeError만 건너뜀
- 시계 동기화는 상태파일(server_time*) 우선, 없으면 선택적으로 HEAD Date 사용
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import json
import os
import re
import urllib.request
import urllib.error

import streamlit as st

# 선택: 자동 새로고침(없어도 동작)
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:
    st_autorefresh = None  # type: ignore


# ============================ 부트스트랩 ============================ #
def _ensure_runs_root_bootstrap() -> None:
    """
    RUNS_ROOT/runs 또는 ./runs 존재 보장(읽기 전용 뷰어 UX 안정화).
    - RUNS_ROOT 지정 시: RUNS_ROOT/runs 생성
    - 미지정 시: ./runs 생성
    생성 실패(OSError)는 치명적 아님(세션 탐색만 수행).
    """
    runs_root_env = os.environ.get("RUNS_ROOT", "").strip()
    try:
        if runs_root_env:
            (Path(runs_root_env).expanduser().resolve() / "runs").mkdir(parents=True, exist_ok=True)
        else:
            Path("./runs").resolve().mkdir(parents=True, exist_ok=True)
    except OSError:
        # 권한/네트워크 드라이브 문제 시에도 뷰어는 계속 진행
        pass


# ============================== 공용 유틸/IO ============================== #

def _get_query_param(name: str) -> str:
    """Streamlit 신/구 API 호환 쿼리 파라미터 읽기."""
    try:
        qp = st.query_params  # type: ignore[attr-defined]
    except AttributeError:
        q = st.experimental_get_query_params()
        return (q.get(name, [""])[0] or "").strip()
    val = qp.get(name, "")
    return val if isinstance(val, str) else str(val)


def _set_query_param(name: str, value: str) -> None:
    """Streamlit 신/구 API 호환 쿼리 파라미터 쓰기."""
    try:
        qp = st.query_params  # type: ignore[attr-defined]
        qp[name] = value  # type: ignore[index]
    except AttributeError:
        st.experimental_set_query_params(**{name: value})


def _list_run_sessions() -> List[Path]:
    """세션 디렉터리 후보(최신순). RUNS_ROOT/runs/* → RUNS_ROOT/* → ./runs/* → ./live_runs/*."""
    roots: List[Path] = []
    env_root = os.environ.get("RUNS_ROOT", "").strip()
    if env_root:
        roots += [Path(env_root) / "runs", Path(env_root)]
    roots += [Path("./runs"), Path("./live_runs")]

    out: List[Path] = []
    for base in roots:
        if not base.exists() or not base.is_dir():
            continue
        for p in base.iterdir():
            if p.is_dir():
                out.append(p.resolve())
    out.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return out


def _pick_session(sessions: List[Path]) -> Optional[Path]:
    if not sessions:
        return None
    wanted = _get_query_param("session")
    label_map = {f"{p.parent.name}/{p.name}": p for p in sessions}
    labels = list(label_map.keys())

    idx = 0
    if wanted:
        for i, lab in enumerate(labels):
            if wanted in lab:
                idx = i
                break

    st.sidebar.subheader("세션 선택")
    label = st.sidebar.selectbox("Session", labels, index=idx)
    _set_query_param("session", label)
    return label_map[label]


def _read_json(path: Path) -> Optional[dict]:
    """JSON 파일 → dict. 파싱 실패/부재는 None."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _read_jsonl_tail(path: Path, limit: int) -> List[dict]:
    """JSONL 꼬리 N행. 손상 라인은 건너뜀."""
    if not path.exists() or limit <= 0:
        return []
    dq: deque[str] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            dq.append(line.rstrip("\n"))
    out: List[dict] = []
    for line in dq:
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _read_text_tail(path: Path, limit_lines: int) -> List[str]:
    """TEXT 꼬리 N행."""
    if not path.exists() or limit_lines <= 0:
        return []
    dq: deque[str] = deque(maxlen=limit_lines)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            dq.append(line.rstrip("\n"))
    return list(dq)


# ============================== 데이터 소스 탐색 ============================== #

def _candidate_status_files(session_dir: Path) -> Dict[str, Path]:
    """관용 상태 파일 후보를 반환(키는 부모/파일명으로 유일화)."""
    cands = [
        session_dir / "status" / "system_status.json",
        session_dir / "outputs" / "system_status.json",
        session_dir / "logs" / "system_status.json",
        session_dir / "broker" / "status.json",
        session_dir / "broker" / "heartbeat.json",
        session_dir / "outputs" / "broker_status.json",
        session_dir / "logs" / "broker_status.json",
    ]
    return {f"{p.parent.name}/{p.name}": p for p in cands if p.exists()}


def _candidate_log_files(session_dir: Path) -> List[Path]:
    """관용 로그 파일 목록(최대 일부)."""
    logs_dir = session_dir / "logs"
    if not logs_dir.exists():
        return []
    out: List[Path] = []
    for p in logs_dir.iterdir():
        if p.is_file() and p.suffix in {".log", ".jsonl"}:
            out.append(p)
    out.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return out[:6]  # 최신 6개


# ============================== 해석/평가 ============================== #

@dataclass
class BrokerStatus:
    provider: str = "—"
    connected: Optional[bool] = None
    authenticated: Optional[bool] = None
    latency_ms: Optional[float] = None
    last_heartbeat: Optional[datetime] = None


@dataclass
class RateLimit:
    remaining: Optional[int] = None
    limit: Optional[int] = None
    reset_at: Optional[datetime] = None  # 서버 기준(가능하면 UTC)


@dataclass
class ClockSkew:
    skew_ms: Optional[int] = None  # server_time - local_time
    source: str = "—"              # 'status-file' | 'HEAD Date' | '—'


def _parse_broker_status(d: dict) -> BrokerStatus:
    """다양한 스키마를 관용 해석."""
    provider = d.get("provider") or d.get("exchange") or d.get("broker") or "—"
    connected = d.get("connected")
    if connected is None:
        connected = d.get("ws_connected") or d.get("rest_connected")
    authenticated = d.get("authenticated") if "authenticated" in d else d.get("auth") or d.get("api_key_valid")
    latency_ms = d.get("latency_ms") or d.get("roundtrip_ms") or d.get("ping_ms")

    hb = d.get("last_heartbeat") or d.get("last_ping") or d.get("ts")
    hb_dt: Optional[datetime] = None
    if isinstance(hb, (int, float)):
        try:
            scale = 1000.0 if float(hb) > 1e12 else 1.0
            hb_dt = datetime.fromtimestamp(float(hb) / scale, tz=timezone.utc)
        except (ValueError, OSError, TypeError):
            hb_dt = None
    elif isinstance(hb, str):
        try:
            hb_dt = datetime.fromisoformat(hb.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            hb_dt = None

    return BrokerStatus(
        provider=str(provider),
        connected=bool(connected) if connected is not None else None,
        authenticated=bool(authenticated) if authenticated is not None else None,
        latency_ms=float(latency_ms) if latency_ms is not None else None,
        last_heartbeat=hb_dt,
    )


def _parse_rate_limit(d: dict) -> RateLimit:
    """rate limit 정보 해석."""
    rl = d.get("rate_limit") or d
    remaining = rl.get("remaining") or rl.get("x-ratelimit-remaining") or rl.get("remaining_requests")
    limit = rl.get("limit") or rl.get("x-ratelimit-limit") or rl.get("max_requests")
    reset = rl.get("reset") or rl.get("reset_epoch") or rl.get("x-ratelimit-reset")
    reset_at: Optional[datetime] = None
    if isinstance(reset, (int, float)):
        try:
            reset_at = datetime.fromtimestamp(float(reset), tz=timezone.utc)
        except (ValueError, OSError, TypeError):
            reset_at = None
    elif isinstance(reset, str):
        try:
            reset_at = datetime.fromisoformat(reset.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            reset_at = None
    return RateLimit(
        remaining=int(remaining) if remaining is not None else None,
        limit=int(limit) if limit is not None else None,
        reset_at=reset_at,
    )


def _compute_clock_skew(status: dict, probe_url: Optional[str]) -> ClockSkew:
    """시계 동기화: 상태파일(server_time*) 우선, 없으면 HEAD Date(선택)."""
    server_time = status.get("server_time") or status.get("server_ts") or status.get("exchange_time")
    # 상태 파일 기반
    if isinstance(server_time, (int, float)):
        try:
            scale = 1000.0 if float(server_time) > 1e12 else 1.0
            srv_dt = datetime.fromtimestamp(float(server_time) / scale, tz=timezone.utc)
            skew = int((srv_dt - datetime.now(timezone.utc)).total_seconds() * 1000)
            return ClockSkew(skew_ms=skew, source="status-file")
        except (ValueError, OSError, TypeError):
            pass
    elif isinstance(server_time, str):
        try:
            srv_dt = datetime.fromisoformat(server_time.replace("Z", "+00:00")).astimezone(timezone.utc)
            skew = int((srv_dt - datetime.now(timezone.utc)).total_seconds() * 1000)
            return ClockSkew(skew_ms=skew, source="status-file")
        except ValueError:
            pass

    # HEAD Date 기반(옵션)
    if not probe_url:
        return ClockSkew(skew_ms=None, source="—")
    try:
        req = urllib.request.Request(probe_url, method="HEAD")
        with urllib.request.urlopen(req, timeout=3) as resp:
            date_hdr = resp.headers.get("Date")
        if not date_hdr:
            return ClockSkew(skew_ms=None, source="—")
        srv_dt = datetime.strptime(date_hdr, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
        skew = int((srv_dt - datetime.now(timezone.utc)).total_seconds() * 1000)
        return ClockSkew(skew_ms=skew, source="HEAD Date")
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
        return ClockSkew(skew_ms=None, source="—")


# ============================== 거래 캘린더 ============================== #

@dataclass
class MarketSchedule:
    name: str
    tz: str
    open_hm: Tuple[int, int]
    close_hm: Tuple[int, int]
    weekend_closed: bool
    always_open: bool = False  # Crypto


_MARKETS = {
    "Crypto (24/7)": MarketSchedule("Crypto (24/7)", "UTC", (0, 0), (23, 59), weekend_closed=False, always_open=True),
    "KRX": MarketSchedule("KRX", "Asia/Seoul", (9, 0), (15, 30), weekend_closed=True),
    "NYSE": MarketSchedule("NYSE", "America/New_York", (9, 30), (16, 0), weekend_closed=True),
    "NASDAQ": MarketSchedule("NASDAQ", "America/New_York", (9, 30), (16, 0), weekend_closed=True),
}


def _market_state(mkt: MarketSchedule, now_utc: datetime) -> Tuple[bool, bool, datetime, datetime]:
    """개장중 여부, 시가 도달 여부, 오늘 개시/종료 시각(UTC). (휴일은 미반영/근사)"""
    if mkt.always_open:
        open_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        close_utc = open_utc + timedelta(hours=23, minutes=59)
        return True, True, open_utc, close_utc

    tz = ZoneInfo(mkt.tz)
    now_local = now_utc.astimezone(tz)
    is_weekend = now_local.weekday() >= 5
    open_local = now_local.replace(hour=mkt.open_hm[0], minute=mkt.open_hm[1], second=0, microsecond=0)
    close_local = now_local.replace(hour=mkt.close_hm[0], minute=mkt.close_hm[1], second=0, microsecond=0)
    open_reached = now_local >= open_local
    is_open = open_reached and now_local <= close_local and not (mkt.weekend_closed and is_weekend)
    return is_open, (open_reached and not (mkt.weekend_closed and is_weekend)), open_local.astimezone(timezone.utc), close_local.astimezone(timezone.utc)


# ============================== 로그 이상 탐지 ============================== #

_ERR_PAT = re.compile(r"(ERROR|Exception|Traceback|CRITICAL|Rate.?limit|429|auth|disconnect|timeout)", re.IGNORECASE)


def _scan_logs(session_dir: Path, tail_lines: int = 500) -> Tuple[int, List[str]]:
    """최근 로그에서 이상 키워드 집계."""
    count = 0
    samples: List[str] = []
    for p in _candidate_log_files(session_dir):
        if p.suffix == ".jsonl":
            for rec in _read_jsonl_tail(p, limit=tail_lines):
                msg = json.dumps(rec, ensure_ascii=False)
                if _ERR_PAT.search(msg):
                    count += 1
                    if len(samples) < 8:
                        samples.append(msg[:500])
        else:
            for line in _read_text_tail(p, tail_lines):
                if _ERR_PAT.search(line):
                    count += 1
                    if len(samples) < 8:
                        samples.append(line[:500])
    return count, samples


# ============================== 렌더링 ============================== #

def _badge(text: str, bg: str) -> str:
    return f'<span style="display:inline-block;padding:4px 8px;border-radius:999px;background:{bg};color:#fff;font-weight:600;">{text}</span>'


def _card(title: str, body_md: str) -> None:
    st.markdown(f"### {title}")
    st.markdown(body_md, unsafe_allow_html=True)
    st.markdown("---")


def _render_broker_card(status_files: Dict[str, Path]) -> None:
    # 상태 파일 병합(후보 중 최신 우선)
    picks = [p for p in status_files.values() if ("broker" in str(p) or "system_status" in str(p))]
    picks.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    raw = _read_json(picks[0]) if picks else None
    bs = _parse_broker_status(raw or {})
    rl = _parse_rate_limit(raw or {})

    conn_badge = _badge("연결 OK", "#0f9960") if bs.connected else _badge("연결 불안", "#d33") if bs.connected is not None else _badge("불명", "#999")
    auth_badge = _badge("인증 OK", "#0f9960") if bs.authenticated else _badge("인증 실패", "#d33") if bs.authenticated is not None else _badge("불명", "#999")

    rl_text = "—"
    rl_badge = _badge("정보 없음", "#999")
    if rl.limit is not None and rl.remaining is not None:
        pct = (rl.remaining / max(rl.limit, 1)) * 100.0
        rl_text = f"{rl.remaining}/{rl.limit}"
        color = "#0f9960" if pct >= 50 else "#f0ad4e" if pct >= 10 else "#d33"
        rl_badge = _badge(f"잔여 {pct:.0f}%", color)
    reset_text = rl.reset_at.astimezone(ZoneInfo("Asia/Seoul")).strftime("%H:%M:%S %Z") if rl.reset_at else "—"

    lat_text = f"{bs.latency_ms:.0f} ms" if bs.latency_ms is not None else "—"
    hb_text = bs.last_heartbeat.astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S %Z") if bs.last_heartbeat else "—"

    body = (
        f"{_badge(bs.provider, '#1f77b4')} &nbsp; {conn_badge} &nbsp; {auth_badge}  \n"
        f"**레이트 리밋**: {rl_text} &nbsp; {rl_badge} &nbsp; (리셋 {reset_text})  \n"
        f"**지연**: {lat_text} &nbsp; **하트비트**: {hb_text}"
    )
    _card("브로커 API 상태", body)


def _render_clock_card(status_files: Dict[str, Path]) -> None:
    picks = [p for p in status_files.values() if p.name.split("/")[-1] in {"system_status.json", "status.json", "broker_status.json", "heartbeat.json"}]
    picks.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    raw = _read_json(picks[0]) if picks else {}

    probe_url = os.environ.get("SYNC_PROBE_URL", "").strip() or None
    skew = _compute_clock_skew(raw or {}, probe_url)

    if skew.skew_ms is None:
        badge = _badge("측정 불가", "#999")
        detail = "—"
    else:
        ms = abs(skew.skew_ms)
        color = "#0f9960" if ms < 300 else "#f0ad4e" if ms < 1500 else "#d33"
        sign = "빠름" if skew.skew_ms < 0 else "느림"
        badge = _badge(f"{ms} ms ({sign})", color)
        detail = f"기준: {skew.source}"

    _card("시계 동기화", f"{badge}  \n{detail}")


def _render_calendar_card() -> None:
    market_name = st.sidebar.selectbox("시장(근사)", list(_MARKETS.keys()), index=0)
    mkt = _MARKETS[market_name]
    is_open, open_reached, open_utc, close_utc = _market_state(mkt, datetime.now(timezone.utc))

    open_local = open_utc.astimezone(ZoneInfo(mkt.tz))
    close_local = close_utc.astimezone(ZoneInfo(mkt.tz))
    now_local = datetime.now(ZoneInfo(mkt.tz))

    open_badge = _badge("개장중", "#0f9960") if is_open else _badge("폐장", "#d33")
    reach_badge = _badge("시가 도달", "#0f9960") if open_reached else _badge("시가 전", "#f0ad4e")
    body = (
        f"{_badge(mkt.name, '#1f77b4')} &nbsp; {open_badge} &nbsp; {reach_badge}  \n"
        f"영업시간(근사): {open_local.strftime('%H:%M')}–{close_local.strftime('%H:%M')} ({mkt.tz})  \n"
        f"현재 시각: {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )
    _card("거래 캘린더/시가", body)


def _render_log_card(session_dir: Path) -> None:
    n_err, samples = _scan_logs(session_dir, tail_lines=400)
    color = "#0f9960" if n_err == 0 else "#f0ad4e" if n_err <= 5 else "#d33"
    badge = _badge(f"이상 {n_err}건", color)
    _card("이상 로그", f"{badge}")
    if samples:
        with st.expander("샘플(최대 8개)"):
            for s in samples:
                st.code(s, language="text")


def _render_overall_banner() -> None:
    st.markdown("## 📟 System Tracker — 실시간 상태")
    st.caption("※ 휴일/특수세션/브로커별 규정은 반영되지 않은 근사치일 수 있습니다.")


# ============================== 메인 ============================== #

def _sidebar() -> Tuple[Optional[Path], int]:
    st.sidebar.header("⚙️ 설정")
    _ensure_runs_root_bootstrap()
    sessions = _list_run_sessions()
    session_dir = _pick_session(sessions)
    refresh_ms = st.sidebar.slider("자동 새로고침(ms)", min_value=2000, max_value=10000, value=3000, step=500)
    st.sidebar.caption("RUNS_ROOT, SYNC_PROBE_URL(HEAD Date 기준 시간) 지원")
    return session_dir, int(refresh_ms)


def main() -> None:
    st.set_page_config(page_title="System Tracker", layout="wide")
    _render_overall_banner()

    session_dir, refresh_ms = _sidebar()
    if not session_dir:
        st.error("세션 디렉터리를 찾을 수 없습니다. runs/* 또는 live_runs/* 하위를 확인하세요.")
        st.stop()

    if st_autorefresh:
        st_autorefresh(interval=refresh_ms, key="auto-refresh-system")

    status_files = _candidate_status_files(session_dir)

    cols = st.columns(3)
    with cols[0]:
        _render_broker_card(status_files)
    with cols[1]:
        _render_clock_card(status_files)
    with cols[2]:
        _render_calendar_card()

    _render_log_card(session_dir)

    st.caption(f"Session: {session_dir}")


if __name__ == "__main__":
    main()

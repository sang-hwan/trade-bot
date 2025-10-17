# monitoring/trade_log_viewer.py
# ------------------------------------------------------------
# 매매 로그 시각화 대시보드 (Streamlit)
# - 실시간 trades.jsonl을 우선 읽고, 없으면 trades.csv로 폴백
# - 1~10초 자동 새로고침(슬라이더 조절)
# 실행 예:
#   streamlit run monitoring/trade_log_viewer.py --server.port 8501
#   # 또는 runs 루트를 환경변수로 지정
#   RUNS_ROOT=/path/to/runs streamlit run monitoring/trade_log_viewer.py
# ------------------------------------------------------------
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# 선택 모듈(없으면 폴백)
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:
    st_autorefresh = None  # type: ignore


def _list_run_sessions(runs_root: Path) -> list[Path]:
    """runs/* 하위 세션 디렉터리를 최신 수정순으로 반환."""
    if not runs_root.exists():
        return []
    items = [p for p in runs_root.iterdir() if p.is_dir()]
    items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return items


def _read_jsonl(path: Path, limit: Optional[int] = None) -> list[dict]:
    """JSON Lines 파일에서 최근 limit행까지 읽어 리스트[dict] 반환."""
    rows: list[dict] = []
    if not path or not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                # 실시간 append 중 손상된 라인은 건너뜀
                continue
    if limit is not None and len(rows) > limit:
        rows = rows[-limit:]
    return rows


def _find_file(base: Path, name: str) -> Optional[Path]:
    """
    관용 경로 우선순위로 파일 탐색:
      1) base/name
      2) base/outputs/name
      3) base/logs/name
    """
    for cand in (base / name, base / "outputs" / name, base / "logs" / name):
        if cand.exists():
            return cand
    return None


def _coalesce(*vals):
    """앞에서부터 None이 아닌 첫 값을 반환."""
    for v in vals:
        if v is not None:
            return v
    return None


def _to_local_ts(ts: str, tz: str = "Asia/Seoul") -> str:
    """UTC 문자열을 로컬표시(YYYY-mm-dd HH:MM:SS)로 변환. 실패 시 원본 반환."""
    try:
        s = pd.to_datetime(ts, utc=True)
        s = s.tz_convert(tz)
        return s.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


# -------------------------- 페이지 설정 --------------------------
st.set_page_config(page_title="Trade Log Viewer", layout="wide")
st.sidebar.title("⚙️ 설정")

# runs 루트: 기본 ./runs, 환경변수 RUNS_ROOT 우선(없으면 생성)
runs_root = Path(os.environ.get("RUNS_ROOT", "./runs")).expanduser().resolve()
runs_root.mkdir(parents=True, exist_ok=True)

# 세션 선택: 쿼리스트링 ?session=... 우선
qs = st.query_params
qs_session = qs.get("session")
session_qs = qs_session if isinstance(qs_session, str) else None

sessions = _list_run_sessions(runs_root)
session_labels = [p.name for p in sessions]
default_idx = 0
if session_qs and session_qs in session_labels:
    default_idx = session_labels.index(session_qs)

sel_session = st.sidebar.selectbox(
    "세션 선택 (runs/*)",
    session_labels or ["<세션 없음>"],
    index=default_idx if session_labels else 0,
)
session_dir = runs_root / sel_session if session_labels else None

refresh_ms = st.sidebar.slider("자동 새로고침 (ms)", 1000, 10000, 3000, 500)
max_rows = st.sidebar.slider("표시할 최대 행 수", 50, 5000, 500, 50)
st.sidebar.caption("환경변수 RUNS_ROOT 로 runs 루트 경로 지정 가능")

# 자동 새로고침 (선택 모듈 존재 시)
if st_autorefresh:
    st_autorefresh(interval=refresh_ms, key="trade_log_auto_refresh")

# -------------------------- 본문 --------------------------
st.title("🧾 Trade Log Viewer")

if not session_dir or not session_dir.exists():
    st.warning("runs/* 세션 폴더를 찾지 못했습니다. RUNS_ROOT 또는 세션을 확인하세요.")
    st.stop()

# -------------------------- 입력 소스 선택 --------------------------
# 1) JSONL 우선: ./, outputs/, logs/ 순으로 탐색
used_path: Optional[Path] = None
jl_path = _find_file(session_dir, "trades.jsonl")
rows = _read_jsonl(jl_path, limit=max_rows) if jl_path else []
df = pd.DataFrame(rows)
if not df.empty:
    used_path = jl_path

# 2) 비어있으면 CSV 폴백
if df.empty:
    csv_path = _find_file(session_dir, "trades.csv")
    if csv_path:
        try:
            df = pd.read_csv(csv_path)
            used_path = csv_path
        except (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError):
            df = pd.DataFrame()

if df.empty:
    st.info("표시할 거래가 없습니다. (trades.jsonl/csv 미존재 또는 비어있음)")
    st.stop()

# 표준 컬럼 정리
df["ts"] = df.apply(lambda r: _coalesce(r.get("ts"), r.get("ts_utc"), r.get("time")), axis=1)
df["ts_local"] = df["ts"].map(lambda s: _to_local_ts(s) if isinstance(s, str) else s)
for col in ("qty", "price", "commission"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
if "qty" in df.columns and "price" in df.columns:
    df["notional"] = df["qty"].fillna(0) * df["price"].fillna(0)

# KPI
total_trades = len(df)
if "side" in df.columns:
    buys = int((df["side"] == "buy").sum())
    sells = int((df["side"] == "sell").sum())
elif "reason" in df.columns:
    # reason 텍스트에서 buy/sell 키워드 추정
    _reason = df["reason"].astype(str)
    buys = int(_reason.str.contains("buy", case=False, na=False).sum())
    sells = int(_reason.str.contains("sell", case=False, na=False).sum())
else:
    buys = sells = 0

notional_sum = float(pd.to_numeric(df.get("notional", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
commission_sum = float(pd.to_numeric(df.get("commission", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("총 거래수", f"{total_trades:,}")
c2.metric("매수/매도", f"{buys:,} / {sells:,}")
c3.metric("총 체결 금액(절대)", f"{notional_sum:,.2f}")
c4.metric("수수료 합계", f"{commission_sum:,.4f}")

st.divider()

# 필터
with st.expander("🔎 필터"):
    symbols = sorted(df.get("symbol", pd.Series(dtype=str)).dropna().unique().tolist())
    sides = sorted(df.get("side", pd.Series(dtype=str)).dropna().unique().tolist())
    reasons = sorted(df.get("reason", pd.Series(dtype=str)).dropna().unique().tolist())

    sel_symbols = st.multiselect("심볼", symbols, default=[])
    sel_sides = st.multiselect("사이드", sides, default=[])
    sel_reasons = st.multiselect("사유", reasons, default=[])

    if sel_symbols:
        df = df[df["symbol"].isin(sel_symbols)]
    if sel_sides and "side" in df:
        df = df[df["side"].isin(sel_sides)]
    if sel_reasons and "reason" in df:
        df = df[df["reason"].isin(sel_reasons)]

# 정렬: 최신순
if "ts" in df.columns:
    try:
        df["_ts_sort"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.sort_values("_ts_sort", ascending=False).drop(columns=["_ts_sort"])
    except Exception:
        df = df.sort_values("ts", ascending=False)

show_cols = [
    c
    for c in [
        "ts_local",
        "ts",
        "symbol",
        "side",
        "qty",
        "price",
        "notional",
        "commission",
        "reason",
        "broker",
        "tif",
        "order_id",
    ]
    if c in df.columns
]

st.subheader("실시간 매매 로그")
st.dataframe(df[show_cols].head(max_rows), use_container_width=True)

st.caption(f"세션: {sel_session} · 경로: {used_path if used_path else 'N/A'} · {len(df)}행 중 상위 {min(max_rows, len(df))}행 표시")

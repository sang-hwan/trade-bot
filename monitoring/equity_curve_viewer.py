# monitoring/equity_curve_viewer.py
"""
수익 실시간 시각화 (Streamlit)

공개 API/계약
- 입력: RUNS_ROOT/runs/<session>/equity_curve.csv (기본), 또는 기존 live_runs/<session>/equity_curve.csv
- 출력: 누적 수익률, 일중 P/L, 드로우다운 라인차트 + 목표 대비 편차 배지, KPI 카드
- 자동 갱신: 3~5초 간격(슬라이더 조정), streamlit-autorefresh 존재 시 사용

수식/타이밍
- 누적수익률: R_t = Equity_t / Equity_0 - 1
- 일중 P/L: P_t = Equity_t - Equity_first_of_local_day(Asia/Seoul)
- 드로우다운: DD_t = Equity_t / max_{s≤t}(Equity_s) - 1, MDD = min_t(DD_t)
- CSV는 실시간 append를 가정: 최근 N행만 읽기 가능
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo
import io
import os

import pandas as pd
import streamlit as st

# 선택 모듈(없으면 폴백)
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
    실패(OSError)는 치명적 아님(세션 탐색만 수행).
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


# ============================== 공용 유틸 ============================== #

def _get_query_param(name: str) -> str:
    """Streamlit 신/구 API 호환 쿼리 파라미터 읽기."""
    try:
        qp = st.query_params  # type: ignore[attr-defined]
        val = qp.get(name, "")
        return val if isinstance(val, str) else str(val)
    except Exception:
        q = st.experimental_get_query_params()
        return (q.get(name, [""])[0] or "").strip()


def _set_query_param(name: str, value: str) -> None:
    """Streamlit 신/구 API 호환 쿼리 파라미터 쓰기."""
    try:
        qp = st.query_params  # type: ignore[attr-defined]
        qp[name] = value  # type: ignore[index]
    except Exception:
        st.experimental_set_query_params(**{name: value})


def _list_run_sessions() -> List[Path]:
    """
    세션 디렉터리 후보(최신순).
    우선순위: RUNS_ROOT/runs/* → RUNS_ROOT/* → ./runs/* → ./live_runs/*.
    equity_curve.csv 존재 디렉터리만 채택.
    """
    roots: List[Path] = []
    runs_root_env = os.environ.get("RUNS_ROOT", "").strip()
    if runs_root_env:
        roots += [Path(runs_root_env) / "runs", Path(runs_root_env)]
    roots += [Path("./runs"), Path("./live_runs")]

    seen: Dict[str, Path] = {}
    for base in roots:
        if not base.exists() or not base.is_dir():
            continue
        for p in base.iterdir():
            if not p.is_dir():
                continue
            if any((p / sub / "equity_curve.csv").exists() for sub in ("", "outputs", "logs")):
                seen[str(p.resolve())] = p.resolve()

    sessions = list(seen.values())

    def _mtime(d: Path) -> float:
        for cand in [d / "equity_curve.csv", d / "outputs" / "equity_curve.csv", d / "logs" / "equity_curve.csv"]:
            if cand.exists():
                return cand.stat().st_mtime
        return d.stat().st_mtime

    sessions.sort(key=_mtime, reverse=True)
    return sessions


def _find_equity_csv(path: Path) -> Optional[Path]:
    """세션 디렉터리에서 equity_curve.csv 경로 탐색."""
    for cand in [path / "equity_curve.csv", path / "outputs" / "equity_curve.csv", path / "logs" / "equity_curve.csv"]:
        if cand.exists():
            return cand
    return None


def _read_csv_tail(path: Path, limit_rows: Optional[int]) -> pd.DataFrame:
    """
    CSV에서 최근 N행만 로드(헤더 유지).
    - 대용량에서 파싱 비용 절감.
    """
    if not path.exists():
        return pd.DataFrame()
    if not limit_rows or limit_rows <= 0:
        return pd.read_csv(path)

    header = ""
    dq: deque[str] = deque(maxlen=limit_rows)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        for line in f:
            dq.append(line)
    buf = header + "".join(dq)
    return pd.read_csv(io.StringIO(buf))


def _to_local_str(ts: str, tz: str = "Asia/Seoul") -> str:
    """ISO-8601Z/naive 문자열을 로컬 시간 문자열로."""
    try:
        if ts.endswith("Z"):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            dt = pd.to_datetime(ts, utc=True).to_pydatetime()
        return dt.astimezone(ZoneInfo(tz)).strftime("%Y-%m-%d %H:%M:%S %Z")
    except (ValueError, TypeError, AttributeError):
        return ts


# ============================== 데이터 해석 ============================== #

@dataclass
class CurveColumns:
    ts: str
    equity: str


def _resolve_columns(df: pd.DataFrame) -> Optional[CurveColumns]:
    """관용 컬럼명 해석: ts/시간, equity/잔고 총액."""
    ts_candidates = ["ts", "time", "timestamp", "datetime"]
    eq_candidates = ["equity", "total_equity", "final_equity", "Equity", "equity_base"]
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    eq_col = next((c for c in eq_candidates if c in df.columns), None)
    if not ts_col or not eq_col:
        return None
    return CurveColumns(ts=ts_col, equity=eq_col)


def _prepare_curve(df: pd.DataFrame, cols: CurveColumns) -> pd.DataFrame:
    """타임스탬프 UTC 정규화, 정렬, 결측 제거."""
    out = df[[cols.ts, cols.equity]].copy()
    out[cols.ts] = pd.to_datetime(out[cols.ts], utc=True, errors="coerce")
    out = out.dropna(subset=[cols.ts, cols.equity]).sort_values(cols.ts).reset_index(drop=True)
    return out


def _compute_series(curve: pd.DataFrame, cols: CurveColumns) -> dict:
    """
    파생 시계열:
    - cum_ret(누적 수익률), intraday_pl(일중 P/L), dd(드로우다운), mdd(스칼라), ts_local(표시용)
    """
    eq = pd.to_numeric(curve[cols.equity], errors="coerce")
    eq0 = float(eq.iloc[0])
    if eq0 == 0:  # 0으로 시작하는 비정상 데이터 방어
        return dict(
            cum_ret=pd.Series([0.0] * len(eq), index=curve.index, dtype="float64"),
            dd=pd.Series([0.0] * len(eq), index=curve.index, dtype="float64"),
            mdd=0.0,
            intraday_pl=pd.Series(dtype="float64"),
            ts_local=curve[cols.ts].dt.tz_convert("Asia/Seoul"),
        )

    cum_ret = eq / eq0 - 1.0
    roll_max = eq.cummax().replace(0, pd.NA).astype("float64")
    dd = (eq / roll_max) - 1.0
    dd = dd.fillna(0.0)
    mdd = float(dd.min())

    ts_local = curve[cols.ts].dt.tz_convert("Asia/Seoul")
    today_mask = ts_local.dt.date == ts_local.iloc[-1].date()
    eq_today = eq.loc[today_mask]
    intraday_pl = pd.Series(dtype="float64")
    if not eq_today.empty:
        intraday_pl = eq_today - eq_today.iloc[0]

    return dict(cum_ret=cum_ret, dd=dd, mdd=mdd, intraday_pl=intraday_pl, ts_local=ts_local)


# ============================== 페이지/사이드바 ============================== #

def _pick_session(sessions: List[Path]) -> Optional[Path]:
    if not sessions:
        return None
    wanted = _get_query_param("session")
    label_map = {f"{p.parent.name}/{p.name}": p for p in sessions}
    labels = list(label_map.keys())

    default_idx = 0
    if wanted:
        for i, lab in enumerate(labels):
            if wanted in lab:
                default_idx = i
                break

    st.sidebar.subheader("세션 선택")
    label = st.sidebar.selectbox("Session", labels, index=default_idx)
    _set_query_param("session", label)
    return label_map[label]


def _sidebar() -> Tuple[Optional[Path], int, int, float]:
    st.sidebar.header("⚙️ 설정")
    _ensure_runs_root_bootstrap()
    sessions = _list_run_sessions()
    session_dir = _pick_session(sessions)
    refresh_ms = st.sidebar.slider("자동 새로고침(ms)", min_value=1000, max_value=10000, value=3000, step=500)
    limit_rows = st.sidebar.number_input("최대 로드 행수", min_value=2000, max_value=200000, value=30000, step=2000)
    target_pct = st.sidebar.number_input("목표 누적수익률(%)", min_value=-100.0, max_value=500.0, value=10.0, step=0.5)
    st.sidebar.caption("RUNS_ROOT 환경변수로 세션 루트를 지정할 수 있습니다. (예: /mnt/logs)")
    return session_dir, int(refresh_ms), int(limit_rows), float(target_pct)


# ============================== 메인 뷰 ============================== #

def _kpis_and_badge(curve: pd.DataFrame, cols: CurveColumns, derived: dict, target_pct: float) -> None:
    last_equity = float(curve[cols.equity].iloc[-1])
    cum_ret_pct = float(derived["cum_ret"].iloc[-1] * 100.0)
    intraday_pl_last = float(derived["intraday_pl"].iloc[-1]) if not derived["intraday_pl"].empty else 0.0
    mdd_pct = float(derived["dd"].min() * -100.0)  # 양수로 표시

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("누적 수익률", f"{cum_ret_pct:,.2f}%")
    c2.metric("일중 P/L", f"{intraday_pl_last:,.0f}")
    c3.metric("MDD", f"{mdd_pct:,.2f}%")
    c4.metric("최신 잔고", f"{last_equity:,.0f}")

    deviation = cum_ret_pct - target_pct
    badge_color = "#0f9960" if deviation >= 0 else "#d33"
    badge_text = "상회" if deviation >= 0 else "하회"
    st.markdown(
        f'<div style="display:inline-block;padding:6px 10px;border-radius:999px;background:{badge_color};'
        f'color:white;font-weight:600;">목표 {target_pct:.2f}% 대비 {badge_text} ({deviation:+.2f}%p)</div>',
        unsafe_allow_html=True,
    )

    last_ts = curve[cols.ts].iloc[-1].isoformat().replace("+00:00", "Z")
    st.caption(f"마지막 갱신: {_to_local_str(last_ts)}")


def _plot_lines(curve: pd.DataFrame, cols: CurveColumns, derived: dict) -> None:
    import matplotlib.pyplot as plt  # seaborn 미사용, 색상 지정 안 함

    fig1, ax1 = plt.subplots()
    ax1.plot(curve[cols.ts], derived["cum_ret"] * 100.0)
    ax1.set_title("누적 수익률(%)")
    ax1.set_xlabel("시간")
    ax1.set_ylabel("%")
    st.pyplot(fig1)

    if not derived["intraday_pl"].empty:
        today_ts = curve[cols.ts].loc[derived["intraday_pl"].index]
        fig2, ax2 = plt.subplots()
        ax2.plot(today_ts, derived["intraday_pl"])
        ax2.set_title("일중 P/L")
        ax2.set_xlabel("시간")
        ax2.set_ylabel("금액")
        st.pyplot(fig2)
    else:
        st.info("오늘 데이터가 충분하지 않아 일중 P/L을 표시할 수 없습니다.")

    fig3, ax3 = plt.subplots()
    ax3.plot(curve[cols.ts], derived["dd"] * 100.0)
    ax3.set_title("드로우다운(%)")
    ax3.set_xlabel("시간")
    ax3.set_ylabel("%")
    st.pyplot(fig3)


def main() -> None:
    st.set_page_config(page_title="Equity Curve Viewer", layout="wide")
    st.title("📈 Equity Curve Viewer — 수익 실시간 시각화")

    session_dir, refresh_ms, limit_rows, target_pct = _sidebar()
    if not session_dir:
        st.error("세션 디렉터리를 찾을 수 없습니다. runs/* 또는 live_runs/* 하위를 확인하세요.")
        st.stop()

    if st_autorefresh:
        st_autorefresh(interval=refresh_ms, key="auto-refresh-equity")

    csv_path = _find_equity_csv(session_dir)
    if not csv_path:
        st.warning("equity_curve.csv 파일을 찾지 못했습니다.")
        st.stop()

    df = _read_csv_tail(csv_path, limit_rows=limit_rows)
    if df.empty:
        st.warning("equity_curve.csv 가 비어 있거나 읽을 수 없습니다.")
        st.stop()

    cols = _resolve_columns(df)
    if not cols:
        st.error("equity_curve.csv의 컬럼을 해석할 수 없습니다. (필수: ts/time/timestamp, equity/total_equity 등)")
        st.stop()

    curve = _prepare_curve(df, cols)
    if curve.empty or len(curve) < 2:
        st.warning("표시할 시계열이 부족합니다.")
        st.stop()

    derived = _compute_series(curve, cols)
    _kpis_and_badge(curve, cols, derived, target_pct)
    _plot_lines(curve, cols, derived)

    st.caption(f"Session: {session_dir}")
    st.caption("※ 실시간 파일 갱신/지연으로 일시적으로 불연속/결측이 보일 수 있습니다.")


if __name__ == "__main__":
    main()

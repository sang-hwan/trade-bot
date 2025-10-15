# monitoring/portfolio_viewer.py
"""
포트폴리오 실시간 시각화 (Streamlit)

공개 API/계약
- 입력: RUNS_ROOT/runs/<session>/fills.jsonl (기본), 또는 기존 live_runs/<session>/fills.jsonl
- 출력: 화면상 포트폴리오 비중 차트 + 종목별 손익표, KPI 카드
- 기준통화: USD/KRW 중 택일. 교차 통화는 USDKRW 환율로 환산

타이밍/예외
- JSONL은 실시간 append를 가정: 손상 라인은 건너뜀(json.JSONDecodeError만 무시)
- 외부 시세(Upbit/야후)는 실패 시 일부 값 NaN/— 표시
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import urllib.request
import urllib.error
import os

import pandas as pd
import streamlit as st

# 선택 모듈: 없으면 해당 기능만 생략
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:
    st_autorefresh = None  # type: ignore


# ============================== 데이터 모델 ============================== #

@dataclass
class Fill:
    symbol: str
    side: str  # "BUY" | "SELL"
    qty: float
    price: float
    fee: float = 0.0
    price_ccy: Optional[str] = None  # "USD" | "KRW"
    ts: Optional[str] = None         # ISO-8601 UTC


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_cost: float = 0.0
    price_ccy: str = ""


# ============================== 공용 유틸 ============================== #

def _to_local(ts: Optional[str], tz: str = "Asia/Seoul") -> str:
    """ISO-8601Z → 로컬 문자열. 파싱 실패 시 원문 반환."""
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ZoneInfo(tz))
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except (ValueError, TypeError):
        return ts


def _get_query_param(name: str) -> str:
    """Streamlit 신/구 API 호환 쿼리 파라미터 가져오기."""
    try:
        qp = st.query_params  # type: ignore[attr-defined]
        val = qp.get(name, "")
        return val if isinstance(val, str) else str(val)
    except Exception:
        q = st.experimental_get_query_params()
        return (q.get(name, [""])[0] or "").strip()


def _set_query_param(name: str, value: str) -> None:
    """Streamlit 신/구 API 호환 쿼리 파라미터 설정."""
    try:
        qp = st.query_params  # type: ignore[attr-defined]
        qp[name] = value  # type: ignore[index]
    except Exception:
        st.experimental_set_query_params(**{name: value})


def _list_run_sessions() -> List[Path]:
    """
    세션 디렉터리 후보를 나열(최근 수정순).
    우선순위: RUNS_ROOT/runs/* → RUNS_ROOT/* → ./runs/* → ./live_runs/*.
    fills.jsonl 보유 디렉터리만 채택.
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
            if (p / "fills.jsonl").exists() or (p / "logs" / "fills.jsonl").exists() or (p / "fill" / "fills.jsonl").exists():
                seen[str(p.resolve())] = p.resolve()

    sessions = list(seen.values())

    def _mtime(d: Path) -> float:
        for cand in [d / "fills.jsonl", d / "logs" / "fills.jsonl", d / "fill" / "fills.jsonl"]:
            if cand.exists():
                return cand.stat().st_mtime
        return d.stat().st_mtime

    sessions.sort(key=_mtime, reverse=True)
    return sessions


def _find_jsonl(path: Path) -> Optional[Path]:
    """세션 디렉터리 내 fills.jsonl 탐색."""
    for cand in [path / "fills.jsonl", path / "logs" / "fills.jsonl", path / "fill" / "fills.jsonl"]:
        if cand.exists():
            return cand
    return None


def _read_jsonl_tail(path: Path, limit: Optional[int] = None) -> List[dict]:
    """
    JSONL에서 최근 N행을 리스트로 반환.
    - json.JSONDecodeError만 무시(실시간 append로 중간 라인 손상 가능)
    """
    if not path.exists():
        return []
    if limit is None or limit <= 0:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    else:
        from collections import deque as _deque
        dq: _deque[str] = _deque(maxlen=limit)
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                dq.append(line.rstrip("\n"))
        lines = list(dq)

    out: List[dict] = []
    for line in lines:
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_fills(session_dir: Path, limit: Optional[int]) -> List[Fill]:
    """fills.jsonl → Fill 리스트(관용 키 인식)."""
    target = _find_jsonl(session_dir)
    if not target:
        return []

    out: List[Fill] = []
    for rec in _read_jsonl_tail(target, limit=limit):
        symbol = rec.get("symbol") or rec.get("ticker") or rec.get("code")
        side = (rec.get("side") or rec.get("action") or "").upper()
        qty = rec.get("qty") or rec.get("quantity") or rec.get("filled_qty") or 0
        price = rec.get("price") or rec.get("avg_price") or rec.get("fill_price") or 0
        fee = rec.get("fee") or rec.get("commission") or 0
        ts = rec.get("ts") or rec.get("time") or rec.get("timestamp")
        price_ccy = rec.get("price_ccy") or rec.get("currency")

        if not symbol or side not in {"BUY", "SELL"}:
            continue
        if not price_ccy:
            price_ccy = infer_price_ccy(str(symbol))

        out.append(
            Fill(
                symbol=str(symbol),
                side=side,
                qty=float(qty),
                price=float(price),
                fee=float(fee),
                price_ccy=str(price_ccy),
                ts=str(ts) if ts else None,
            )
        )
    return out


def reconstruct_positions(fills: List[Fill]) -> Dict[str, Position]:
    """
    체결 로그로 보유 포지션 재구성.
    - BUY: (기존가치 + 신규원가 + 수수료) / 신규수량
    - SELL: 수량 감소, 0이 되면 평단 0
    """
    pos: Dict[str, Position] = {}
    for f in fills:
        p = pos.get(f.symbol)
        if not p:
            p = Position(symbol=f.symbol, price_ccy=f.price_ccy)
            pos[f.symbol] = p

        if f.side == "BUY":
            new_qty = p.qty + f.qty
            if new_qty <= 0:
                p.qty = 0.0
                p.avg_cost = 0.0
            else:
                total_cost = p.avg_cost * p.qty + f.price * f.qty + f.fee
                p.avg_cost = total_cost / new_qty
                p.qty = new_qty
        else:  # SELL
            p.qty = max(0.0, p.qty - f.qty)
            if p.qty == 0.0:
                p.avg_cost = 0.0

        if not p.price_ccy:
            p.price_ccy = f.price_ccy

    return {k: v for k, v in pos.items() if v.qty > 0}


# ============================== 시세/환율 ============================== #

def infer_price_ccy(symbol: str, default_us_equity_ccy: str = "USD") -> str:
    """심볼로 가격 통화 추정(Upbit KRW- 접두어는 KRW, 그 외 USD)."""
    return "KRW" if str(symbol).upper().startswith("KRW-") else default_us_equity_ccy


def fetch_upbit_prices(symbols: List[str]) -> Dict[str, float]:
    """Upbit KRW-티커 현재가(표준 라이브러리 사용)."""
    syms = [s for s in symbols if s.upper().startswith("KRW-")]
    if not syms:
        return {}
    url = "https://api.upbit.com/v1/ticker?markets=" + ",".join(syms)
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {d["market"]: float(d["trade_price"]) for d in data}
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return {}


def fetch_equity_prices(symbols: List[str]) -> Dict[str, float]:
    """미국 주식 등(USD) 현재가: yfinance가 있을 때만 사용."""
    if yf is None:
        return {}
    out: Dict[str, float] = {}
    for s in (x for x in symbols if not x.upper().startswith("KRW-")):
        try:
            t = yf.Ticker(s)
            px = getattr(getattr(t, "fast_info", None), "last_price", None)
            if px is None:
                h = t.history(period="1d")
                if not h.empty:
                    px = float(h["Close"].iloc[-1])
            if px is not None:
                out[s] = float(px)
        except Exception:
            continue
    return out


def fetch_usdkrw() -> Optional[float]:
    """USD/KRW 환율(yfinance: USDKRW=X). 실패 시 None."""
    if yf is None:
        return None
    try:
        t = yf.Ticker("USDKRW=X")
        px = getattr(getattr(t, "fast_info", None), "last_price", None)
        if px is None:
            h = t.history(period="1d", interval="1h")
            if not h.empty:
                px = float(h["Close"].iloc[-1])
        return float(px) if px is not None else None
    except Exception:
        return None


def convert_value(v: float, from_ccy: str, to_ccy: str, usdkrw: Optional[float]) -> float:
    """USD↔KRW 변환. 동일 통화면 그대로, 환율 없음은 NaN."""
    if from_ccy == to_ccy:
        return v
    if usdkrw is None:
        return float("nan")
    if from_ccy == "USD" and to_ccy == "KRW":
        return v * usdkrw
    if from_ccy == "KRW" and to_ccy == "USD":
        return v / usdkrw
    return float("nan")


# ============================== 집계/표시 ============================== #

def build_table(
    positions: Dict[str, Position],
    prices: Dict[str, float],
    base_ccy: str,
    usdkrw: Optional[float],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for sym, p in positions.items():
        last = prices.get(sym)
        mv = (last * p.qty) if (last is not None) else float("nan")
        upnl = ((last - p.avg_cost) * p.qty) if (last is not None) else float("nan")
        upnl_pct = (((last / p.avg_cost) - 1.0) * 100.0) if (last is not None and p.avg_cost > 0) else float("nan")

        mv_base = convert_value(mv, p.price_ccy, base_ccy, usdkrw) if not pd.isna(mv) else float("nan")
        upnl_base = convert_value(upnl, p.price_ccy, base_ccy, usdkrw) if not pd.isna(upnl) else float("nan")

        # ⚠️ dict() 키워드 인자로는 '%'가 들어간 키를 쓸 수 없음 → 리터럴 사용
        row = {
            "Symbol": sym,
            "Qty": p.qty,
            "AvgCost": f"{p.avg_cost:,.6g} {p.price_ccy}",
            "LastPrice": (f"{last:,.6g} {p.price_ccy}" if last is not None else "—"),
            "UPNL": f"{upnl:,.6g} {p.price_ccy}" if not pd.isna(upnl) else "—",
            "UPNL_%": (round(upnl_pct, 2) if not pd.isna(upnl_pct) else float("nan")),
            "MV": f"{mv:,.6g} {p.price_ccy}" if not pd.isna(mv) else "—",
            "MV_base": (mv_base if not pd.isna(mv_base) else float("nan")),
            "UPNL_base": (upnl_base if not pd.isna(upnl_base) else float("nan")),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    total_mv_base = df["MV_base"].sum(skipna=True)
    df["Weight_%"] = (df["MV_base"] / total_mv_base * 100.0) if total_mv_base > 0 else float("nan")
    show = ["Symbol", "Qty", "AvgCost", "LastPrice", "UPNL", "UPNL_%", "MV", "MV_base", "Weight_%"]
    return df[show].sort_values("Weight_%", ascending=False, na_position="last")


def draw_pie(df: pd.DataFrame, base_ccy: str) -> None:
    import matplotlib.pyplot as plt  # seaborn 미사용, 색상 지정 없음

    sdf = df[["Symbol", "MV_base"]].dropna()
    sdf = sdf[sdf["MV_base"] > 0]
    if sdf.empty:
        st.info("표시할 보유 종목이 없습니다.")
        return

    labels = sdf["Symbol"].tolist()
    weights = sdf["MV_base"].tolist()

    fig, ax = plt.subplots()
    ax.pie(weights, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    ax.set_title(f"포트폴리오 비중 ({base_ccy} 기준)")
    st.pyplot(fig)


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


def _sidebar() -> Tuple[Optional[Path], str, int, float, int, bool]:
    st.sidebar.header("⚙️ 설정")
    sessions = _list_run_sessions()
    session_dir = _pick_session(sessions)
    base_ccy = st.sidebar.selectbox("기준 통화(Base Currency)", ["USD", "KRW"], index=0)
    limit = st.sidebar.number_input("최대 체결 읽기(N행)", min_value=1000, max_value=200000, value=10000, step=1000)
    height = st.sidebar.number_input("표 높이(px)", min_value=320, max_value=1600, value=560, step=40)
    refresh_ms = st.sidebar.slider("자동 새로고침(ms)", min_value=1000, max_value=15000, value=5000, step=500)
    use_autorefresh = st_autorefresh is not None and st.sidebar.checkbox("자동 새로고침 사용", value=True)
    st.sidebar.caption("RUNS_ROOT 환경변수로 루트를 지정할 수 있습니다. (예: /mnt/logs)")
    return session_dir, base_ccy, int(refresh_ms), float(height), int(limit), bool(use_autorefresh)


# ============================== 메인 ============================== #

def main() -> None:
    st.set_page_config(page_title="Portfolio Viewer", layout="wide")
    st.title("📊 Portfolio Viewer — 실시간 포트폴리오 시각화")

    session_dir, base_ccy, refresh_ms, table_height, read_limit, use_autorefresh = _sidebar()

    if not session_dir:
        st.error("세션 디렉터리를 찾을 수 없습니다. runs/* 또는 live_runs/* 하위를 확인하세요.")
        st.stop()

    if use_autorefresh and st_autorefresh:
        st_autorefresh(interval=refresh_ms, key="auto-refresh-portfolio")

    fills = load_fills(session_dir, limit=read_limit)
    if not fills:
        st.warning("fills.jsonl 을 찾지 못했거나 비어 있습니다. 체결 수집 후 다시 확인하세요.")
        st.stop()

    positions = reconstruct_positions(fills)
    if not positions:
        st.info("보유 포지션이 없습니다.")
        st.stop()

    symbols = sorted(positions.keys())
    st.subheader("심볼 필터")
    sel = st.multiselect("표시할 종목을 선택하세요(미선택 시 전체)", options=symbols, default=symbols)
    symbols_view = sel if sel else symbols

    prices: Dict[str, float] = {}
    prices.update(fetch_upbit_prices(symbols_view))
    prices.update(fetch_equity_prices(symbols_view))

    usdkrw = fetch_usdkrw()
    if base_ccy in {"USD", "KRW"} and usdkrw is None and any(infer_price_ccy(s) != base_ccy for s in symbols_view):
        st.warning("USD/KRW 환율을 가져오지 못했습니다. 교차 통화 자산의 MV/UPNL(기준통화)은 NaN으로 표시됩니다.")

    pos_view = {s: positions[s] for s in symbols_view}
    df = build_table(pos_view, prices, base_ccy, usdkrw)

    last_ts = max((f.ts for f in fills if f.ts), default=None)
    total_mv_base = df["MV_base"].sum(skipna=True) if not df.empty else 0.0
    total_upnl_base = df["UPNL_base"].sum(skipna=True) if not df.empty else 0.0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("총 평가액", f"{total_mv_base:,.0f} {base_ccy}")
    c2.metric("총 미실현손익", f"{total_upnl_base:,.0f} {base_ccy}")
    c3.metric("USD/KRW", f"{usdkrw:,.2f}" if usdkrw is not None else "—")
    top_sym, top_w = None, None
    if not df.empty and df["Weight_%"].notna().any():
        top_idx = df["Weight_%"].astype(float).idxmax()
        top_sym = df.loc[top_idx, "Symbol"]
        top_w = df.loc[top_idx, "Weight_%"]
    c4.metric(
        "최대 비중",
        f"{top_sym or '—'}{(' (' + format(top_w, '.2f') + '%)') if (top_w is not None and not pd.isna(top_w)) else ''}",
    )
    c5.metric("마지막 체결", _to_local(last_ts))

    left, right = st.columns([5, 7])
    with left:
        draw_pie(df, base_ccy)
    with right:
        st.subheader("종목별 손익표")
        if df.empty:
            st.info("표시할 데이터가 없습니다.")
        else:
            df_show = df.copy()
            df_show["UPNL_%"] = df_show["UPNL_%"].map(lambda x: f"{x:.2f}%" if not pd.isna(x) else "—")
            df_show["MV_base"] = df_show["MV_base"].map(lambda x: f"{x:,.0f} {base_ccy}" if not pd.isna(x) else "—")
            df_show["Weight_%"] = df_show["Weight_%"].map(lambda x: f"{x:.2f}%" if not pd.isna(x) else "—")
            st.dataframe(df_show, use_container_width=True, height=int(table_height))

    st.caption(f"Session: {session_dir}")
    st.caption("※ 시세 지연/네트워크 오류 시 일부 값이 NaN/— 로 표시될 수 있습니다. Upbit: KRW-*, 미국주식: USD 가정.")


if __name__ == "__main__":
    main()

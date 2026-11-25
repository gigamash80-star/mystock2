import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf

try:
    import FinanceDataReader as fdr
except ImportError:
    fdr = None

# =====================================================================
# 1. 페이지 기본 설정
# =====================================================================
st.set_page_config(
    page_title="미장 500·국장 500 주간 퀀트 대시보드",
    page_icon="🌎",
    layout="wide",
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


# =====================================================================
# 2. 미장 500대(S&P 500) 및 국장 시총 500대 종목 자동 수집기
# =====================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_us_500_universe(limit: int = 500) -> dict[str, str]:
    # 1순위: FinanceDataReader S&P 500 목록
    if fdr is not None:
        try:
            sp = fdr.StockListing("S&P500")
            if not sp.empty and "Symbol" in sp.columns:
                sp["Symbol"] = sp["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
                name_col = "Name" if "Name" in sp.columns else "Security"
                sp = sp.drop_duplicates(subset=["Symbol"]).head(limit)
                return dict(zip(sp["Symbol"], sp[name_col]))
        except Exception:
            pass

    # 2순위 백업: Wikipedia S&P 500 테이블 직접 파싱
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        try:
            tables = pd.read_html(io.StringIO(resp.text))
            df = tables[0]
            syms = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
            names = df["Security"].astype(str).str.strip()
            return dict(list(zip(syms, names))[:limit])
        except Exception:
            # lxml 미설치 환경 대비 정규식 파싱
            block = resp.text.split('id="constituents"')[-1].split("</table>")[0]
            universe = {}
            for row in block.split("<tr")[1:]:
                tds = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
                if len(tds) >= 2:
                    sym = re.sub(r"<[^>]+>", "", tds[0]).strip().replace(".", "-")
                    name = re.sub(r"<[^>]+>", "", tds[1]).strip()
                    if re.match(r"^[A-Z0-9\-]{1,7}$", sym):
                        universe[sym] = name
                    if len(universe) >= limit:
                        break
            if universe:
                return universe
    except Exception:
        pass

    return {}


@st.cache_data(ttl=86400, show_spinner=False)
def get_kr_500_universe(limit: int = 500) -> dict[str, str]:
    # 1순위: FinanceDataReader KRX 시가총액 상위 종목 (코스피 + 코스닥 통합 시총 순)
    if fdr is not None:
        try:
            krx = fdr.StockListing("KRX")
            if not krx.empty and "Code" in krx.columns:
                krx["Code"] = krx["Code"].astype(str).str.strip()
                krx["Market"] = krx["Market"].astype(str).str.upper()

                # 코스피 및 코스닥(코스닥 글로벌 포함) 보통주만 필터링, 스팩 제외
                cond_mkt = (krx["Market"] == "KOSPI") | (krx["Market"].str.contains("KOSDAQ"))
                cond_common = krx["Code"].str.endswith("0")
                cond_no_spac = ~krx["Name"].astype(str).str.contains("스팩", na=False)
                krx = krx[cond_mkt & cond_common & cond_no_spac]

                if "Marcap" in krx.columns:
                    krx = krx.sort_values(by="Marcap", ascending=False)

                krx = krx.head(limit)
                universe = {}
                for _, r in krx.iterrows():
                    suffix = "KS" if r["Market"] == "KOSPI" else "KQ"
                    universe[f"{r['Code']}.{suffix}"] = str(r["Name"]).strip()
                if universe:
                    return universe
        except Exception:
            pass

    # 2순위 백업: 네이버 금융 시가총액 상위 페이지 파싱 (코스피 75% + 코스닥 25%)
    universe = {}
    kospi_target = int(limit * 0.75)
    kosdaq_target = limit - kospi_target

    for sosok, suffix, target_cnt in [(0, "KS", kospi_target), (1, "KQ", kosdaq_target)]:
        collected = 0
        for page in range(1, 15):
            if collected >= target_cnt:
                break
            try:
                url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"
                resp = requests.get(url, headers=HEADERS, timeout=10)
                resp.encoding = "euc-kr"
                matches = re.findall(
                    r'<a[^>]*href="/item/main\.naver\?code=(\d{6})"[^>]*>([^<]+)</a>',
                    resp.text,
                )
                for code, name in matches:
                    name = name.strip()
                    if not name or "스팩" in name or not code.endswith("0"):
                        continue
                    key = f"{code}.{suffix}"
                    if key not in universe:
                        universe[key] = name
                        collected += 1
                        if collected >= target_cnt:
                            break
            except Exception:
                break

    return universe


# =====================================================================
# 3. 1단계(배치 기술적 분석) + 2단계(병렬 PER·ROE 재무 분석) 엔진
# =====================================================================
def safe_float(val) -> float | None:
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def extract_ticker_df(raw_df: pd.DataFrame, sym: str) -> pd.DataFrame | None:
    if raw_df is None or raw_df.empty:
        return None
    if isinstance(raw_df.columns, pd.MultiIndex):
        lvl0 = raw_df.columns.get_level_values(0)
        lvl1 = raw_df.columns.get_level_values(1)
        if sym in lvl0:
            sub = raw_df[sym].copy()
        elif sym in lvl1:
            sub = raw_df.xs(sym, axis=1, level=1).copy()
        else:
            return None
    else:
        sub = raw_df.copy()

    req_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not req_cols.issubset(set(sub.columns)):
        return None

    sub = sub.dropna(subset=["Open", "High", "Low", "Close"])
    if len(sub) < 120:
        return None

    idx = pd.to_datetime(sub.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    sub.index = idx
    return sub


def score_technical(df: pd.DataFrame, sym: str, name: str, market: str) -> dict | None:
    try:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"].fillna(0)

        curr_price = float(close.iloc[-1])
        if curr_price <= 0:
            return None

        df["SMA20"] = close.rolling(20).mean()
        df["SMA60"] = close.rolling(60).mean()
        df["SMA120"] = close.rolling(120).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = (100 - (100 / (1 + rs))).fillna(100)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line

        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            atr = curr_price * 0.03

        curr_sma20 = float(df["SMA20"].iloc[-1])
        curr_sma60 = float(df["SMA60"].iloc[-1])
        curr_sma120 = float(df["SMA120"].iloc[-1])
        curr_rsi = float(df["RSI"].iloc[-1])

        ret_5d = float((curr_price / close.iloc[-6] - 1) * 100)
        ret_20d = float((curr_price / close.iloc[-21] - 1) * 100)

        vol_20_mean = float(volume.iloc[-20:].mean())
        vol_ratio = float(volume.iloc[-5:].mean() / vol_20_mean) if vol_20_mean > 0 else 0.0

        tech_score = 0
        signals = []

        # [A] 추세 (25점)
        if curr_price > curr_sma20:
            tech_score += 10
        if curr_sma20 > curr_sma60:
            tech_score += 10
            signals.append("중기정배열")
        if curr_price > curr_sma120:
            tech_score += 5

        # [B] 모멘텀 - RSI & MACD (20점)
        if 50 <= curr_rsi <= 70:
            tech_score += 10
            signals.append(f"RSI양호({curr_rsi:.0f})")
        elif 35 <= curr_rsi < 50 and ret_5d > 0:
            tech_score += 8
            signals.append("눌림목반등")
        elif curr_rsi > 70:
            tech_score += 3

        if macd_line.iloc[-1] > signal_line.iloc[-1]:
            tech_score += 7
            if macd_hist.iloc[-1] > macd_hist.iloc[-2]:
                tech_score += 3
                signals.append("MACD확장")
        elif macd_hist.iloc[-1] > macd_hist.iloc[-2] > macd_hist.iloc[-3]:
            tech_score += 5

        # [C] 거래량 수급 및 상대강도 (25점)
        if ret_5d > 0 and vol_ratio >= 1.3:
            tech_score += 15
            signals.append(f"거래량급증({vol_ratio:.1f}x)")
        elif ret_5d > 0 and vol_ratio >= 1.0:
            tech_score += 10
        elif vol_ratio >= 1.1:
            tech_score += 4

        if ret_20d > 0:
            tech_score += 5
        if 0 < ret_5d <= 12:
            tech_score += 5
        elif ret_5d > 12:
            tech_score += 2

        return {
            "market": market,
            "symbol": sym,
            "name": name,
            "label": f"{name} ({sym})",
            "price": curr_price,
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "rsi": curr_rsi,
            "vol_ratio": vol_ratio,
            "tech_score": tech_score,
            "stop_loss": curr_price - (1.5 * atr),
            "target_price": curr_price + (2.5 * atr),
            "tech_signals": signals,
            "history": df.tail(140),
        }
    except Exception:
        return None


def enrich_fundamental(item: dict) -> dict:
    sym = item["symbol"]
    per = None
    roe = None
    try:
        info = yf.Ticker(sym).info or {}
        per = safe_float(info.get("trailingPE"))
        if per is None:
            per = safe_float(info.get("forwardPE"))

        roe_raw = safe_float(info.get("returnOnEquity"))
        roe = roe_raw * 100 if roe_raw is not None else None
        if roe is None:
            eps = safe_float(info.get("trailingEps"))
            bv = safe_float(info.get("bookValue"))
            if eps is not None and bv is not None and bv > 0:
                roe = (eps / bv) * 100
    except Exception:
        pass

    fund_score = 0
    signals = list(item["tech_signals"])

    # PER 점수 (15점)
    if per is None:
        fund_score += 7
    elif 0 < per <= 15:
        fund_score += 15
        signals.append(f"저PER({per:.1f})")
    elif 15 < per <= 30:
        fund_score += 12
    elif 30 < per <= 50:
        fund_score += 7
    elif per > 50:
        fund_score += 3

    # ROE 점수 (15점)
    if roe is None:
        fund_score += 7
    elif roe >= 20:
        fund_score += 15
        signals.append(f"고ROE({roe:.1f}%)")
    elif 12 <= roe < 20:
        fund_score += 12
    elif 5 <= roe < 12:
        fund_score += 7
    elif 0 < roe < 5:
        fund_score += 3

    enriched = item.copy()
    enriched["per"] = per
    enriched["roe"] = roe
    enriched["fund_score"] = fund_score
    enriched["total_score"] = item["tech_score"] + fund_score
    enriched["signals"] = ", ".join(signals) if signals else "중립"
    return enriched


def run_two_stage_scan(
    universe: dict[str, str],
    market: str,
    fund_pool_size: int,
    priority_symbols: set[str],
    progress_bar,
) -> tuple[list[dict], int]:
    symbols = list(universe.keys())
    total_syms = len(symbols)
    batch_size = 100
    tech_results = []

    # 1단계: 100개 단위 일봉 일괄 다운로드 및 전 종목 기술적 채점
    for start in range(0, total_syms, batch_size):
        batch = symbols[start : start + batch_size]
        end_idx = min(start + batch_size, total_syms)
        progress_bar.progress(
            end_idx / (total_syms * 1.4),
            text=f"[{market}] 1단계 일봉 차트 전수 스캔 중... ({end_idx}/{total_syms}종목)",
        )
        try:
            raw_df = yf.download(
                tickers=batch if len(batch) > 1 else batch[0],
                period="1y",
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            for sym in batch:
                sub_df = extract_ticker_df(raw_df, sym)
                if sub_df is not None:
                    scored = score_technical(sub_df, sym, universe[sym], market)
                    if scored is not None:
                        tech_results.append(scored)
        except Exception:
            continue

    scanned_count = len(tech_results)
    if not tech_results:
        return [], 0

    # 기술적 점수 + 20일 상대강도 순 정렬 후 상위 후보군 추출 (직접 추가한 관심종목은 무조건 포함)
    tech_results.sort(key=lambda x: (x["tech_score"], x["ret_20d"]), reverse=True)
    top_candidates = tech_results[:fund_pool_size]
    picked_syms = {x["symbol"] for x in top_candidates}
    for item in tech_results[fund_pool_size:]:
        if item["symbol"] in priority_symbols and item["symbol"] not in picked_syms:
            top_candidates.append(item)

    # 2단계: 상위 후보군 멀티스레드 PER·ROE 재무 분석
    final_items = []
    cand_total = len(top_candidates)
    done_cnt = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(enrich_fundamental, c): c for c in top_candidates}
        for fut in as_completed(futures):
            done_cnt += 1
            progress_bar.progress(
                min(0.71 + 0.29 * (done_cnt / max(cand_total, 1)), 1.0),
                text=f"[{market}] 2단계 상위 유망주 PER·ROE 정밀 분석 중... ({done_cnt}/{cand_total}종목)",
            )
            try:
                final_items.append(fut.result())
            except Exception:
                pass

    return final_items, scanned_count


# =====================================================================
# 4. 실시간 재무 필터링 및 시각화 함수
# =====================================================================
def apply_filters(
    items: list[dict],
    use_filter: bool,
    max_per: float,
    min_roe: float,
    exclude_deficit: bool,
    allow_missing: bool,
) -> list[dict]:
    if not use_filter:
        return sorted(items, key=lambda x: (x["total_score"], x["ret_20d"]), reverse=True)

    filtered = []
    for item in items:
        per = item["per"]
        roe = item["roe"]

        if per is None:
            if not allow_missing:
                continue
        else:
            if exclude_deficit and per <= 0:
                continue
            if per > max_per:
                continue

        if roe is None:
            if not allow_missing:
                continue
        else:
            if roe < min_roe:
                continue

        filtered.append(item)

    return sorted(filtered, key=lambda x: (x["total_score"], x["ret_20d"]), reverse=True)


def build_display_df(items: list[dict], market: str) -> pd.DataFrame:
    rows = []
    for rank, r in enumerate(items, start=1):
        if market == "US":
            p_str = f"${r['price']:,.2f}"
            guide_str = f"손절 ${r['stop_loss']:,.2f} / 목표 ${r['target_price']:,.2f}"
        else:
            p_str = f"{r['price']:,.0f}원"
            guide_str = f"손절 {r['stop_loss']:,.0f}원 / 목표 {r['target_price']:,.0f}원"

        rows.append(
            {
                "순위": rank,
                "종목명": r["label"],
                "총점(100)": r["total_score"],
                "기술(70)": r["tech_score"],
                "재무(30)": r["fund_score"],
                "현재가": p_str,
                "주간수익률": f"{r['ret_5d']:+.2f}%",
                "PER(배)": f"{r['per']:.1f}" if r["per"] is not None else "N/A",
                "ROE(%)": f"{r['roe']:.1f}%" if r["roe"] is not None else "N/A",
                "RSI(14)": round(r["rsi"], 1),
                "거래량비": f"{r['vol_ratio']:.2f}배",
                "핵심시그널": r["signals"],
                "주간대응가이드": guide_str,
            }
        )
    return pd.DataFrame(rows)


def plot_stock_chart(item: dict) -> go.Figure:
    df = item["history"]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(
            f"{item['label']} 일봉 & 이동평균선 / ATR 대응선",
            "거래량 (Volume)",
            "RSI (14)",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="일봉",
        ),
        row=1,
        col=1,
    )
    for col_name, color, label in [
        ("SMA20", "#FF9800", "20일선"),
        ("SMA60", "#2196F3", "60일선"),
        ("SMA120", "#9C27B0", "120일선"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col_name],
                mode="lines",
                line=dict(color=color, width=1.5),
                name=label,
            ),
            row=1,
            col=1,
        )

    fig.add_hline(
        y=item["target_price"],
        line_dash="dash",
        line_color="#00C853",
        annotation_text="1차 목표가",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=item["stop_loss"],
        line_dash="dot",
        line_color="#FF1744",
        annotation_text="손절 기준선",
        row=1,
        col=1,
    )

    colors = np.where(df["Close"] >= df["Open"], "#EF5350", "#42A5F5")
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="거래량"),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            mode="lines",
            line=dict(color="#26A69A", width=1.8),
            name="RSI(14)",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=3, col=1)

    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(
        height=720,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# =====================================================================
# 5. 사이드바 설정 및 메인 실행 흐름
# =====================================================================
st.title("🌎 미장 500대 · 국장 500대 주간 종목 추천 대시보드")

with st.sidebar:
    st.header("📡 시장 스캔 범위 설정")
    scan_limit = st.select_slider(
        "시장별 스캔 기업 규모 (시총 상위)",
        options=[100, 200, 300, 500],
        value=500,
    )
    fund_pool_size = st.slider(
        "2단계 PER·ROE 정밀검증 후보 수",
        min_value=30,
        max_value=150,
        value=60,
        step=10,
        help="1단계 전체 차트 스캔 후 상위 몇 개 종목의 재무제표를 정밀 조회할지 결정합니다.",
    )
    extra_us = st.text_input("미국 관심 티커 강제 포함 (쉼표 구분)", placeholder="예: IONQ, RKLB")
    extra_kr = st.text_input("국내 관심 코드 강제 포함 (.KS/.KQ)", placeholder="예: 035720.KS")

    force_rescan = st.button("🚀 전체 종목 새로 스캔하기", type="primary", use_container_width=True)

    st.divider()
    st.header("🎯 추천 및 재무 필터 (즉시 반영)")
    top_n = st.slider("시장별 최종 추천 종목 수 (Top N)", min_value=5, max_value=30, value=10)
    use_fund_filter = st.toggle("PER·ROE 재무 필터 켜기", value=True)
    max_per = st.slider(
        "최대 허용 PER (배 이하)",
        min_value=5,
        max_value=120,
        value=60,
        step=5,
        disabled=not use_fund_filter,
    )
    min_roe = st.slider(
        "최소 요구 ROE (% 이상)",
        min_value=-20,
        max_value=40,
        value=5,
        step=1,
        disabled=not use_fund_filter,
    )
    exclude_deficit = st.checkbox("적자 기업 제외 (양수 PER만 허용)", value=True, disabled=not use_fund_filter)
    allow_missing = st.checkbox("재무 지표 미제공(N/A) 종목 포함", value=True, disabled=not use_fund_filter)

# 스캔 파라미터 키 확인 (슬라이더만 조작할 때는 재다운로드 방지)
scan_key = f"{scan_limit}_{fund_pool_size}_{extra_us.strip()}_{extra_kr.strip()}"

if force_rescan or st.session_state.get("scan_key") != scan_key:
    with st.status(
        f"🔍 미장 {scan_limit}개 + 국장 {scan_limit}개 기업 데이터를 불러와 분석 중입니다... (약 45~60초 소요)",
        expanded=True,
    ) as status:
        st.write("1️⃣ S&P 500 및 국내 시가총액 상위 종목 리스트 확보 중...")
        us_univ = get_us_500_universe(scan_limit).copy()
        kr_univ = get_kr_500_universe(scan_limit).copy()

        prio_us = set()
        if extra_us.strip():
            for s in [x.strip().upper() for x in extra_us.split(",") if x.strip()]:
                us_univ.setdefault(s, s)
                prio_us.add(s)

        prio_kr = set()
        if extra_kr.strip():
            for s in [x.strip().upper() for x in extra_kr.split(",") if x.strip()]:
                kr_univ.setdefault(s, s)
                prio_kr.add(s)

        pbar = st.progress(0, text="데이터 수집 준비 중...")
        us_items, us_cnt = run_two_stage_scan(us_univ, "US", fund_pool_size, prio_us, pbar)
        kr_items, kr_cnt = run_two_stage_scan(kr_univ, "KR", fund_pool_size, prio_kr, pbar)
        pbar.empty()

        st.session_state["scan_key"] = scan_key
        st.session_state["us_items"] = us_items
        st.session_state["kr_items"] = kr_items
        st.session_state["us_cnt"] = us_cnt
        st.session_state["kr_cnt"] = kr_cnt
        st.session_state["scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        status.update(
            label=f"✅ 분석 완료! (미장 {us_cnt}종목 · 국장 {kr_cnt}종목 전수 스캔 완료)",
            state="complete",
            expanded=False,
        )

us_raw = st.session_state.get("us_items", [])
kr_raw = st.session_state.get("kr_items", [])
us_cnt = st.session_state.get("us_cnt", 0)
kr_cnt = st.session_state.get("kr_cnt", 0)
scan_time = st.session_state.get("scan_time", "-")

st.caption(
    f"최근 스캔 시각: {scan_time} | "
    f"1단계 전수 차트 스캔: 미장 {us_cnt}종목 + 국장 {kr_cnt}종목 (총 {us_cnt + kr_cnt}종목) → "
    f"2단계 재무 정밀 검증: 시장별 상위 {fund_pool_size}종목"
)

us_filtered = apply_filters(us_raw, use_fund_filter, max_per, min_roe, exclude_deficit, allow_missing)[:top_n]
kr_filtered = apply_filters(kr_raw, use_fund_filter, max_per, min_roe, exclude_deficit, allow_missing)[:top_n]

tab_us, tab_kr = st.tabs(
    [
        f"🇺🇸 미국 S&P {scan_limit} 추천 (Top {len(us_filtered)})",
        f"🇰🇷 한국 시총 {scan_limit} 추천 (Top {len(kr_filtered)})",
    ]
)


def render_market_tab(items: list[dict], market: str, scanned_total: int):
    if not items:
        st.warning("현재 설정된 PER·ROE 필터 조건을 만족하는 종목이 없습니다. 좌측 사이드바에서 조건을 완화해 보세요.")
        return

    df_view = build_display_df(items, market)
    st.subheader(f"🏆 {scanned_total}개 기업 중 주간 하이브리드 추천 Top {len(items)}")
    st.dataframe(df_view, use_container_width=True, hide_index=True)

    csv_bytes = df_view.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=f"📥 {market} 추천 Top {len(items)} CSV 다운로드",
        data=csv_bytes,
        file_name=f"weekly_{market}_top{len(items)}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("🔍 종목별 상세 차트 및 스코어 분석")
    selected_label = st.selectbox(
        "상세 일봉 차트와 손절·목표 라인을 확인할 종목을 선택하세요",
        options=[x["label"] for x in items],
        key=f"select_{market}",
    )
    target_item = next(x for x in items if x["label"] == selected_label)

    c1, c2, c3, c4 = st.columns(4)
    price_txt = f"${target_item['price']:,.2f}" if market == "US" else f"{target_item['price']:,.0f}원"
    per_txt = f"{target_item['per']:.1f}배" if target_item["per"] is not None else "N/A"
    roe_txt = f"{target_item['roe']:.1f}%" if target_item["roe"] is not None else "N/A"
    guide_txt = (
        f"${target_item['stop_loss']:,.2f} / ${target_item['target_price']:,.2f}"
        if market == "US"
        else f"{target_item['stop_loss']:,.0f}원 / {target_item['target_price']:,.0f}원"
    )

    c1.metric("현재가 (주간 등락)", price_txt, f"{target_item['ret_5d']:+.2f}%")
    c2.metric(
        "종합 점수 (100점)",
        f"{target_item['total_score']}점",
        f"기술 {target_item['tech_score']} + 재무 {target_item['fund_score']}",
    )
    c3.metric("PER / ROE", f"{per_txt} / {roe_txt}", target_item["signals"])
    c4.metric("손절선 / 1차 목표가", guide_txt)

    st.plotly_chart(plot_stock_chart(target_item), use_container_width=True)


with tab_us:
    render_market_tab(us_filtered, "US", us_cnt)

with tab_kr:
    render_market_tab(kr_filtered, "KR", kr_cnt)
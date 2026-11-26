import io
import re
import time
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
    page_title="미장 500·국장 500 전설의 트레이더 퀀트 대시보드",
    page_icon="🔥",
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
    if fdr is not None:
        try:
            sp = fdr.StockListing("S&P500")
            if not sp.empty and "Symbol" in sp.columns:
                sp["Symbol"] = (
                    sp["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
                )
                name_col = "Name" if "Name" in sp.columns else "Security"
                sp = sp.drop_duplicates(subset=["Symbol"]).head(limit)
                return dict(zip(sp["Symbol"], sp[name_col]))
        except Exception:
            pass

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
    if fdr is not None:
        try:
            krx = fdr.StockListing("KRX")
            if not krx.empty and "Code" in krx.columns:
                krx["Code"] = krx["Code"].astype(str).str.strip()
                krx["Market"] = krx["Market"].astype(str).str.upper()

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
# 3. 캔들(봉) 패턴 & 전설적 트레이더 차트 분석 엔진
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


def detect_candle_patterns(df: pd.DataFrame, lookback: int = 30) -> tuple[list[dict], list[str], int]:
    """
    최근 lookback 일봉에서 클래식 캔들(봉) 패턴을 탐지하고,
    최근 3거래일 내 발생한 유효 시그널을 점수화(최대 12점, 하락 경고 시 감점)합니다.
    """
    n = len(df)
    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    vols = df["Volume"].fillna(0).to_numpy(dtype=float)
    vol_ma20 = pd.Series(vols).rolling(20).mean().to_numpy(dtype=float)
    dates = df.index

    markers = []
    recent_bull_tags = []
    recent_bear_tags = []
    best_bull_score = 0
    bear_penalty = 0

    start_pos = max(21, n - lookback)

    for i in range(start_pos, n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        po, ph, pl, pc = opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1]

        rng = max(h - l, 1e-9)
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        is_bull = c > o
        is_bear = c < o

        p_rng = max(ph - pl, 1e-9)
        p_body = abs(pc - po)
        p_is_bull = pc > po
        p_is_bear = pc < po

        v_ok = np.isfinite(vol_ma20[i]) and vol_ma20[i] > 0 and vols[i] >= vol_ma20[i] * 1.2
        is_recent_3 = i >= n - 3
        is_recent_2 = i >= n - 2

        bull_found = None
        bull_pts = 0
        bear_found = None

        # 1) 상승 장악형 (Bullish Engulfing)
        if (
            p_is_bear
            and (p_body / p_rng >= 0.25)
            and is_bull
            and (o <= pc * 1.003)
            and (c >= po)
            and (body > p_body * 1.1)
        ):
            bull_found = "상승장악형"
            bull_pts = 12

        # 2) 샛별형 (Morning Star - 3봉 바닥 반전)
        if bull_found is None and i >= 2:
            o2, h2, l2, c2 = opens[i - 2], highs[i - 2], lows[i - 2], closes[i - 2]
            r2 = max(h2 - l2, 1e-9)
            b2 = abs(c2 - o2)
            if (
                c2 < o2
                and (b2 / r2 >= 0.45)
                and ((c2 - o2) / max(o2, 1e-9) <= -0.015)
                and (p_body / p_rng <= 0.35)
                and is_bull
                and (body / rng >= 0.45)
                and (c >= (o2 + c2) / 2.0)
            ):
                bull_found = "샛별형"
                bull_pts = 12

        # 3) 거래량 동반 장대양봉 (Power Marubozu)
        if (
            bull_found is None
            and is_bull
            and (body / rng >= 0.70)
            and ((c / max(pc, 1e-9) - 1.0) >= 0.03)
            and v_ok
        ):
            bull_found = "장대양봉"
            bull_pts = 12

        # 4) 망치형 / 핀바 (Hammer - 최근 5일 저점 부근 아래꼬리 반등)
        local_low_5 = lows[max(0, i - 5) : i].min()
        if (
            bull_found is None
            and (lower_wick >= max(body, rng * 0.05) * 2.0)
            and (lower_wick / rng >= 0.55)
            and (upper_wick <= rng * 0.22)
            and (body / rng <= 0.35)
            and (l <= local_low_5 * 1.02)
        ):
            bull_found = "망치형(핀바)"
            bull_pts = 9

        # 5) 하락 경고: 고점권 유성형 (Shooting Star) & 하락 장악형 (Bearish Engulfing)
        local_high_10 = highs[max(0, i - 10) : i].max()
        if (
            (upper_wick >= max(body, rng * 0.05) * 2.0)
            and (upper_wick / rng >= 0.55)
            and (lower_wick <= rng * 0.22)
            and (body / rng <= 0.35)
            and (h >= local_high_10 * 0.99)
        ):
            bear_found = "유성형(매도경고)"
        elif (
            p_is_bull
            and (p_body / p_rng >= 0.25)
            and is_bear
            and (o >= pc * 0.997)
            and (c <= po)
            and (body > p_body * 1.1)
            and (h >= local_high_10 * 0.98)
        ):
            bear_found = "하락장악형"

        if bull_found:
            markers.append(
                {"date": dates[i], "price": l * 0.985, "name": bull_found, "type": "bull"}
            )
            if is_recent_3:
                if bull_found not in recent_bull_tags:
                    recent_bull_tags.append(bull_found)
                best_bull_score = max(best_bull_score, bull_pts)

        if bear_found:
            markers.append(
                {"date": dates[i], "price": h * 1.015, "name": bear_found, "type": "bear"}
            )
            if is_recent_2:
                if bear_found not in recent_bear_tags:
                    recent_bear_tags.append(bear_found)
                bear_penalty = 6

    # 최근 3일에 뚜렷한 특수 봉이 없어도 견조한 양봉 우세 시 소폭 가점
    if best_bull_score == 0:
        bull_days = sum(1 for k in range(n - 3, n) if closes[k] > opens[k])
        if bull_days >= 2 and closes[-1] > closes[-4]:
            best_bull_score = 4

    candle_score = max(0, best_bull_score - bear_penalty)
    tags = recent_bull_tags + [f"⚠️{t}" for t in recent_bear_tags]
    return markers, tags, candle_score


def score_technical_and_patterns(df: pd.DataFrame, sym: str, name: str, market: str) -> dict | None:
    try:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"].fillna(0)
        n = len(df)

        curr_price = float(close.iloc[-1])
        if curr_price <= 0:
            return None

        # 이동평균선 (미너비니 트렌드 템플릿용 50·150·200일선 + 단기 20·60일선)
        df["SMA20"] = close.rolling(20).mean()
        df["SMA50"] = close.rolling(50).mean()
        df["SMA60"] = close.rolling(60).mean()
        df["SMA150"] = close.rolling(150).mean()
        df["SMA200"] = close.rolling(200).mean()

        # RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = (100 - (100 / (1 + rs))).fillna(100)

        # MACD(12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line

        # ATR(14)
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            atr = curr_price * 0.03

        curr_sma20 = float(df["SMA20"].iloc[-1])
        curr_sma50 = float(df["SMA50"].iloc[-1])
        curr_sma60 = float(df["SMA60"].iloc[-1])
        curr_sma150 = safe_float(df["SMA150"].iloc[-1])
        curr_sma200 = safe_float(df["SMA200"].iloc[-1])
        prev_sma200 = safe_float(df["SMA200"].iloc[-21]) if n >= 221 else None
        curr_rsi = float(df["RSI"].iloc[-1])

        ret_5d = float((curr_price / close.iloc[-6] - 1) * 100)
        ret_20d = float((curr_price / close.iloc[-21] - 1) * 100)

        vol_20_mean = float(volume.iloc[-20:].mean())
        vol_ratio = float(volume.iloc[-5:].mean() / vol_20_mean) if vol_20_mean > 0 else 0.0

        # 52주 최고가 / 최저가 (최근 252거래일 기준)
        win_52w = df.tail(252)
        high_52w = float(win_52w["High"].max())
        low_52w = float(win_52w["Low"].min())
        from_high_pct = (curr_price / high_52w - 1) * 100 if high_52w > 0 else -100.0
        from_low_pct = (curr_price / low_52w - 1) * 100 if low_52w > 0 else 0.0

        # ==============================================================
        # [영역 1] 전설적 트레이더 차트 구조 (35점 만점)
        # ==============================================================
        legend_score = 0
        chart_tags = []
        is_minervini_stage2 = False

        # 1-A. 마크 미너비니 2단계(Stage 2) 트렌드 템플릿 (15점)
        if (
            curr_sma150 is not None
            and curr_sma200 is not None
            and prev_sma200 is not None
            and curr_price > curr_sma50 > curr_sma150 > curr_sma200
            and curr_sma200 > prev_sma200
        ):
            legend_score += 15
            is_minervini_stage2 = True
            chart_tags.append("미너비니2단계")
        else:
            if curr_price > curr_sma20:
                legend_score += 4
            if curr_sma20 > curr_sma60:
                legend_score += 4
            if curr_sma150 is not None and curr_price > curr_sma150:
                legend_score += 4
            elif curr_price > curr_sma60:
                legend_score += 2

        # 1-B. 윌리엄 오닐 / 미너비니 52주 고저점 위치 (10점)
        if from_low_pct >= 30.0 and from_high_pct >= -25.0:
            legend_score += 7
            if from_high_pct >= -10.0:
                legend_score += 3
                chart_tags.append("52주신고가권")
        elif from_high_pct >= -20.0:
            legend_score += 4

        # 1-C. 니콜라스 다바스 20일 박스권 돌파 / 미너비니 VCP 변동성 축소 (10점)
        highs_arr = high.to_numpy(dtype=float)
        closes_arr = close.to_numpy(dtype=float)
        vols_arr = volume.to_numpy(dtype=float)
        vol_ma20_arr = volume.rolling(20).mean().to_numpy(dtype=float)

        is_box_breakout = False
        for pos in range(n - 3, n):
            prev_20_high = highs_arr[pos - 20 : pos].max()
            if (
                closes_arr[pos] > prev_20_high
                and np.isfinite(vol_ma20_arr[pos])
                and vols_arr[pos] >= vol_ma20_arr[pos] * 1.2
            ):
                is_box_breakout = True
                break

        # VCP: 최근 10일 일중 변동폭 & 거래량이 직전 30일 대비 눈에 띄게 수축 + 50일선 위
        daily_range_pct = (high - low) / close.replace(0, np.nan)
        recent_rng = float(daily_range_pct.iloc[-10:].mean())
        prior_rng = float(daily_range_pct.iloc[-40:-10].mean())
        recent_vol10 = float(volume.iloc[-10:].mean())
        prior_vol30 = float(volume.iloc[-40:-10].mean())
        is_vcp = (
            curr_price > curr_sma50
            and np.isfinite(recent_rng)
            and np.isfinite(prior_rng)
            and prior_rng > 0
            and recent_rng <= prior_rng * 0.78
            and prior_vol30 > 0
            and recent_vol10 <= prior_vol30 * 0.90
        )

        if is_box_breakout:
            legend_score += 10
            chart_tags.append("박스돌파")
        elif is_vcp:
            legend_score += 8
            chart_tags.append("VCP수축")

        # ==============================================================
        # [영역 2] 캔들(봉) 패턴 & 포켓 피벗 수급 (25점 만점)
        # ==============================================================
        candle_markers, candle_tags, candle_pts = detect_candle_patterns(df, lookback=35)

        # 포켓 피벗 (최근 3일 중 양봉일 거래량이 직전 10거래일 최대 음봉 거래량을 상회)
        opens_arr = df["Open"].to_numpy(dtype=float)
        is_pocket_pivot = False
        for pos in range(n - 3, n):
            if closes_arr[pos] > closes_arr[pos - 1] and closes_arr[pos] >= opens_arr[pos]:
                down_vols = [
                    vols_arr[j]
                    for j in range(pos - 10, pos)
                    if closes_arr[j] < closes_arr[j - 1]
                ]
                if down_vols and vols_arr[pos] > max(down_vols) and closes_arr[pos] >= curr_sma50 * 0.98:
                    is_pocket_pivot = True
                    break

        supply_pts = 0
        if is_pocket_pivot:
            supply_pts = 13
            chart_tags.append("포켓피벗")
        elif ret_5d > 0 and vol_ratio >= 1.3:
            supply_pts = 9
            chart_tags.append(f"거래량급증({vol_ratio:.1f}x)")
        elif ret_5d > 0 and vol_ratio >= 1.0:
            supply_pts = 5

        candle_supply_score = candle_pts + supply_pts

        # ==============================================================
        # [영역 3] 단기 모멘텀 & 과열 방지 (15점 만점)
        # ==============================================================
        momentum_score = 0
        if 50 <= curr_rsi <= 70:
            momentum_score += 5
        elif 35 <= curr_rsi < 50 and ret_5d > 0:
            momentum_score += 4

        if macd_line.iloc[-1] > signal_line.iloc[-1]:
            momentum_score += 3
            if macd_hist.iloc[-1] > macd_hist.iloc[-2]:
                momentum_score += 2
        elif macd_hist.iloc[-1] > macd_hist.iloc[-2] > macd_hist.iloc[-3]:
            momentum_score += 2

        if ret_20d > 0:
            momentum_score += 2
        if 0 < ret_5d <= 12:
            momentum_score += 3
        elif ret_5d > 12:
            momentum_score += 1  # 단기 과열 추격 주의

        tech_total_75 = legend_score + candle_supply_score + momentum_score
        all_pattern_tags = chart_tags + candle_tags

        return {
            "market": market,
            "symbol": sym,
            "name": name,
            "label": f"{name} ({sym})",
            "price": curr_price,
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "from_high_pct": from_high_pct,
            "rsi": curr_rsi,
            "vol_ratio": vol_ratio,
            "legend_score": legend_score,
            "candle_supply_score": candle_supply_score,
            "momentum_score": momentum_score,
            "tech_total": tech_total_75,
            "is_minervini": is_minervini_stage2,
            "has_bull_candle": any(not t.startswith("⚠️") for t in candle_tags) or is_pocket_pivot or is_box_breakout,
            "pattern_tags": all_pattern_tags,
            "candle_markers": candle_markers,
            "stop_loss": curr_price - (1.5 * atr),
            "target_price": curr_price + (2.5 * atr),
            "history": df.tail(150),
        }
    except Exception:
        return None


# =====================================================================
# 4. 2단계 재무 조회 안정화 (12시간 캐싱 + 3회 재시도 + 호출 속도 제어)
# =====================================================================
@st.cache_data(ttl=43200, show_spinner=False)
def fetch_fundamental_cached(sym: str) -> tuple[float | None, float | None]:
    for attempt in range(3):
        try:
            time.sleep(0.35 + attempt * 0.65)
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

            if per is not None or roe is not None:
                return per, roe
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return None, None


def enrich_fundamental(item: dict) -> dict:
    per, roe = fetch_fundamental_cached(item["symbol"])
    fund_score = 0
    tags = list(item["pattern_tags"])

    # PER (12점) — 미수신 시 과도한 기본 점수 방지(2점)
    if per is None:
        fund_score += 2
    elif 0 < per <= 15:
        fund_score += 12
        tags.append(f"저PER({per:.1f})")
    elif 15 < per <= 30:
        fund_score += 10
    elif 30 < per <= 50:
        fund_score += 6
    elif per > 50:
        fund_score += 2

    # ROE (13점) — 미수신 시 과도한 기본 점수 방지(2점)
    if roe is None:
        fund_score += 2
    elif roe >= 20:
        fund_score += 13
        tags.append(f"고ROE({roe:.1f}%)")
    elif 12 <= roe < 20:
        fund_score += 10
    elif 5 <= roe < 12:
        fund_score += 6
    elif 0 < roe < 5:
        fund_score += 2

    enriched = item.copy()
    enriched["per"] = per
    enriched["roe"] = roe
    enriched["fund_score"] = fund_score
    enriched["total_score"] = item["tech_total"] + fund_score
    enriched["signals"] = ", ".join(tags) if tags else "중립"
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

    # 1단계: 18개월 일봉 배치 다운로드 (200일선 기울기 및 52주 신고가 정밀 계산용)
    for start in range(0, total_syms, batch_size):
        batch = symbols[start : start + batch_size]
        end_idx = min(start + batch_size, total_syms)
        progress_bar.progress(
            end_idx / (total_syms * 1.45),
            text=f"[{market}] 1단계 전설 차트·봉 패턴 전수 스캔 중... ({end_idx}/{total_syms}종목)",
        )
        try:
            raw_df = yf.download(
                tickers=batch if len(batch) > 1 else batch[0],
                period="18mo",
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            for sym in batch:
                sub_df = extract_ticker_df(raw_df, sym)
                if sub_df is not None:
                    scored = score_technical_and_patterns(sub_df, sym, universe[sym], market)
                    if scored is not None:
                        tech_results.append(scored)
        except Exception:
            continue

    scanned_count = len(tech_results)
    if not tech_results:
        return [], 0

    tech_results.sort(key=lambda x: (x["tech_total"], x["ret_20d"]), reverse=True)
    top_candidates = tech_results[:fund_pool_size]
    picked_syms = {x["symbol"] for x in top_candidates}
    for item in tech_results[fund_pool_size:]:
        if item["symbol"] in priority_symbols and item["symbol"] not in picked_syms:
            top_candidates.append(item)

    # 2단계: 야후 차단 방지를 위해 동시 스레드를 2개로 제한하여 안정적 PER·ROE 조회
    final_items = []
    cand_total = len(top_candidates)
    done_cnt = 0
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(enrich_fundamental, c): c for c in top_candidates}
        for fut in as_completed(futures):
            done_cnt += 1
            progress_bar.progress(
                min(0.69 + 0.31 * (done_cnt / max(cand_total, 1)), 1.0),
                text=f"[{market}] 2단계 상위 유망주 PER·ROE 안정화 검증 중... ({done_cnt}/{cand_total}종목)",
            )
            try:
                final_items.append(fut.result())
            except Exception:
                pass

    return final_items, scanned_count


# =====================================================================
# 5. 실시간 필터링 및 표·차트 시각화
# =====================================================================
def apply_filters(
    items: list[dict],
    only_minervini: bool,
    only_pattern: bool,
    use_fund_filter: bool,
    max_per: float,
    min_roe: float,
    exclude_deficit: bool,
    allow_missing: bool,
) -> list[dict]:
    filtered = []
    for item in items:
        if only_minervini and not item["is_minervini"]:
            continue
        if only_pattern and not item["has_bull_candle"]:
            continue

        if use_fund_filter:
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
                "전설차트(35)": r["legend_score"],
                "봉·수급(25)": r["candle_supply_score"],
                "모멘텀(15)": r["momentum_score"],
                "재무(25)": r["fund_score"],
                "현재가": p_str,
                "주간수익률": f"{r['ret_5d']:+.2f}%",
                "52주고점대비": f"{r['from_high_pct']:+.1f}%",
                "PER": f"{r['per']:.1f}" if r["per"] is not None else "N/A",
                "ROE": f"{r['roe']:.1f}%" if r["roe"] is not None else "N/A",
                "감지된 차트·봉 시그널": r["signals"],
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
        row_heights=[0.62, 0.19, 0.19],
        subplot_titles=(
            f"{item['label']} 일봉 · 미너비니 이평선(20/50/150/200) · 감지된 봉 패턴",
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

    for col_name, color, label, dash in [
        ("SMA20", "#FF9800", "20일선", "solid"),
        ("SMA50", "#2196F3", "50일선", "solid"),
        ("SMA150", "#9C27B0", "150일선", "solid"),
        ("SMA200", "#D32F2F", "200일선", "dash"),
    ]:
        if col_name in df.columns and df[col_name].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col_name],
                    mode="lines",
                    line=dict(color=color, width=1.4, dash=dash),
                    name=label,
                ),
                row=1,
                col=1,
            )

    # 최근 감지된 캔들(봉) 패턴 마커 표시
    bull_m = [m for m in item["candle_markers"] if m["type"] == "bull" and m["date"] in df.index]
    bear_m = [m for m in item["candle_markers"] if m["type"] == "bear" and m["date"] in df.index]

    if bull_m:
        fig.add_trace(
            go.Scatter(
                x=[m["date"] for m in bull_m],
                y=[m["price"] for m in bull_m],
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=11, color="#00C853"),
                text=[m["name"] for m in bull_m],
                textposition="bottom center",
                textfont=dict(size=10, color="#00A152"),
                name="상승 봉 패턴",
            ),
            row=1,
            col=1,
        )

    if bear_m:
        fig.add_trace(
            go.Scatter(
                x=[m["date"] for m in bear_m],
                y=[m["price"] for m in bear_m],
                mode="markers+text",
                marker=dict(symbol="triangle-down", size=11, color="#FF1744"),
                text=[m["name"] for m in bear_m],
                textposition="top center",
                textfont=dict(size=10, color="#D50000"),
                name="고점 매도봉 경고",
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
        height=760,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# =====================================================================
# 6. 사이드바 제어판 및 메인 화면 실행
# =====================================================================
st.title("🔥 미장 500 · 국장 500 전설의 트레이더 & 봉 분석 대시보드")

with st.sidebar:
    st.header("📡 시장 스캔 범위")
    scan_limit = st.select_slider(
        "시장별 스캔 기업 수 (시총 상위)",
        options=[100, 200, 300, 500],
        value=500,
    )
    fund_pool_size = st.slider(
        "2단계 PER·ROE 검증 후보 수",
        min_value=30,
        max_value=100,
        value=50,
        step=10,
        help="클라우드 서버에서는 40~60개 수준이 야후 차단을 피하면서 가장 빠르고 안정적입니다.",
    )
    extra_us = st.text_input("미국 관심 티커 추가 (쉼표 구분)", placeholder="예: IONQ, RKLB")
    extra_kr = st.text_input("국내 관심 코드 추가 (.KS/.KQ)", placeholder="예: 035720.KS")

    force_rescan = st.button("🚀 전체 종목 새로 스캔하기", type="primary", use_container_width=True)

    st.divider()
    st.header("🕯️ 차트·봉·재무 필터 (즉시 반영)")
    top_n = st.slider("시장별 추천 종목 수 (Top N)", min_value=5, max_value=30, value=10)
    only_minervini = st.checkbox("마크 미너비니 2단계(Stage 2) 정배열만 보기", value=False)
    only_pattern = st.checkbox("상승 봉 패턴 · 포켓피벗 · 박스돌파 종목만 보기", value=False)

    st.divider()
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
    exclude_deficit = st.checkbox("적자 기업 제외 (양수 PER만)", value=True, disabled=not use_fund_filter)
    allow_missing = st.checkbox("재무 지표 미제공(N/A) 종목 포함", value=False, disabled=not use_fund_filter)

scan_key = f"{scan_limit}_{fund_pool_size}_{extra_us.strip()}_{extra_kr.strip()}"

if force_rescan or st.session_state.get("scan_key") != scan_key:
    with st.status(
        f"🔍 미장 {scan_limit}개 + 국장 {scan_limit}개 기업을 전설 차트·봉 패턴으로 전수 스캔 중...",
        expanded=True,
    ) as status:
        st.write("1️⃣ S&P 500 및 국내 시가총액 상위 종목 리스트 수집 중...")
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
            label=f"✅ 스캔 완료! (미장 {us_cnt}종목 · 국장 {kr_cnt}종목 분석 완료)",
            state="complete",
            expanded=False,
        )

us_raw = st.session_state.get("us_items", [])
kr_raw = st.session_state.get("kr_items", [])
us_cnt = st.session_state.get("us_cnt", 0)
kr_cnt = st.session_state.get("kr_cnt", 0)
scan_time = st.session_state.get("scan_time", "-")

st.caption(
    f"최근 스캔: {scan_time} | 전수 스캔: 미장 {us_cnt}종목 + 국장 {kr_cnt}종목 | "
    "채점: 전설차트(35) + 봉·포켓피벗(25) + 모멘텀(15) + 재무(25)"
)

us_filtered = apply_filters(
    us_raw, only_minervini, only_pattern, use_fund_filter, max_per, min_roe, exclude_deficit, allow_missing
)[:top_n]
kr_filtered = apply_filters(
    kr_raw, only_minervini, only_pattern, use_fund_filter, max_per, min_roe, exclude_deficit, allow_missing
)[:top_n]

tab_us, tab_kr = st.tabs(
    [
        f"🇺🇸 미국 S&P {scan_limit} 추천 (Top {len(us_filtered)})",
        f"🇰🇷 한국 시총 {scan_limit} 추천 (Top {len(kr_filtered)})",
    ]
)


def render_market_tab(raw_pool: list[dict], items: list[dict], market: str, scanned_total: int):
    # 야후 파이낸스 일시 차단 감지 경고 배너
    if raw_pool:
        valid_fund_cnt = sum(1 for x in raw_pool if x["per"] is not None or x["roe"] is not None)
        success_rate = valid_fund_cnt / len(raw_pool)
        if success_rate < 0.7:
            st.error(
                f"⚠️ 야후 파이낸스 서버의 일시적 호출 제한으로 검증 후보 {len(raw_pool)}개 중 "
                f"{len(raw_pool) - valid_fund_cnt}개 종목의 PER·ROE가 누락되었습니다(수신률 {success_rate*100:.0f}%). "
                "1~2분 뒤 좌측 '🚀 전체 종목 새로 스캔하기'를 눌러주세요. "
                "(임시로 차트 점수만 보려면 좌측 '재무 지표 미제공(N/A) 종목 포함'을 체크하세요.)"
            )

    if not items:
        st.warning(
            "현재 필터 조건을 만족하는 종목이 없습니다. 좌측 사이드바에서 "
            "미너비니/봉 패턴 체크 또는 PER·ROE 조건을 완화해 보세요."
        )
        return

    df_view = build_display_df(items, market)
    st.subheader(f"🏆 {scanned_total}개 기업 중 주간 추천 Top {len(items)}")
    st.dataframe(df_view, use_container_width=True, hide_index=True)

    csv_bytes = df_view.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=f"📥 {market} 추천 Top {len(items)} CSV 다운로드",
        data=csv_bytes,
        file_name=f"weekly_{market}_legend_top{len(items)}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("🔍 종목별 전설 차트 · 캔들 패턴 상세 분석")
    selected_label = st.selectbox(
        "차트와 감지된 봉 패턴 마커를 확인할 종목을 선택하세요",
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
        f"차트 {target_item['legend_score']} + 봉·수급 {target_item['candle_supply_score']} + 모멘텀 {target_item['momentum_score']} + 재무 {target_item['fund_score']}",
    )
    c3.metric("PER / ROE", f"{per_txt} / {roe_txt}", f"52주 고점 대비 {target_item['from_high_pct']:+.1f}%")
    c4.metric("손절선 / 1차 목표가", guide_txt)

    st.info(f"**📌 감지된 핵심 시그널:** {target_item['signals']}")
    st.plotly_chart(plot_stock_chart(target_item), use_container_width=True)


with tab_us:
    render_market_tab(us_raw, us_filtered, "US", us_cnt)

with tab_kr:
    render_market_tab(kr_raw, kr_filtered, "KR", kr_cnt)

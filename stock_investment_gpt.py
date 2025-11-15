import os
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Alpaca Market Data
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest, NewsRequest
from alpaca.data.historical.news import NewsClient
from alpaca.data.timeframe import TimeFrame

# OpenAI client
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- Config ----------
load_dotenv()  # reads .env in the project root

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
SYMBOL = os.getenv("SYMBOL").upper()
PERIOD = os.getenv("PERIOD", "1y").lower()  # accepts '6m', '1y', '2y', etc.

def get_env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in {"1", "true", "yes", "y", "on"}

def get_openai_model() -> str:
    return (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()

ANALYZE_WITH_GPT = get_env_bool("ANALYZE_WITH_GPT", False)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BENCHMARK = "SPY"  # for beta/correlation
NEWS_LIMIT = 5

assert ALPACA_API_KEY and ALPACA_SECRET_KEY, "Missing Alpaca keys in .env"

# ---------- Helpers ----------
def parse_period_to_days(period_str: str) -> int:
    """Convert strings like '6m', '1y', '2y' to an integer day span."""
    if period_str.endswith("d"):
        return int(period_str[:-1])
    if period_str.endswith("w"):
        return int(period_str[:-1]) * 7
    if period_str.endswith("m"):
        return int(period_str[:-1]) * 30
    if period_str.endswith("y"):
        return int(period_str[:-1]) * 365
    # default to 1y if unknown
    return 365

def annualize_vol(daily_ret: pd.Series) -> float:
    return float(daily_ret.std(ddof=1) * np.sqrt(252))

def rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return float(rsi_val.iloc[-1])

def atr(df: pd.DataFrame, period: int = 14) -> float:
    # expects columns: open, high, low, close
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        (df['high'] - df['low']),
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return float(atr_series.iloc[-1])

def pct_change_over(df_close: pd.Series, days: int) -> float:
    if len(df_close) <= days:
        return np.nan
    return float((df_close.iloc[-1] / df_close.iloc[-(days+1)] - 1.0) * 100)

def max_drawdown_pct(close: pd.Series) -> float:
    if close.empty:
        return np.nan
    running_max = close.cummax()
    drawdown = close / running_max - 1.0
    return float(drawdown.min() * 100)

def rolling_sharpe(daily_ret: pd.Series, window: int) -> float:
    r = daily_ret.tail(window)
    if len(r) < 2:
        return np.nan
    std = r.std(ddof=1)
    if std == 0 or pd.isna(std):
        return np.nan
    return float((r.mean() / std) * np.sqrt(252))

@dataclass
class ReturnWindow:
    name: str
    days: int

RETURN_WINDOWS = [
    ReturnWindow("1W", 5),
    ReturnWindow("1M", 21),
    ReturnWindow("3M", 63),
    ReturnWindow("6M", 126),
    ReturnWindow("YTD", None),  # handle separately
    ReturnWindow("1Y", 252),
]

def ytd_change(close: pd.Series) -> float:
    if close.empty:
        return np.nan
    year_start = close[close.index.year == datetime.now(timezone.utc).year]
    if year_start.empty:
        return np.nan
    first = year_start.iloc[0]
    return float((close.iloc[-1] / first - 1.0) * 100)

def moving_average(series: pd.Series, window: int) -> float:
    if len(series) < window:
        return np.nan
    return float(series.tail(window).mean())

def fetch_bars(client, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment='all',  # adjusted for splits/dividends
        feed='iex'        # free feed (works for most US stocks)
    )
    bars = client.get_stock_bars(req)
    df = bars.df
    if df is None or df.empty:
        return pd.DataFrame()
    # When asking for a single symbol, Alpaca still multi-indexes (symbol, timestamp)
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)
    df = df.tz_convert("UTC").sort_index()
    df = df.rename(columns=str.lower)[['open','high','low','close','volume']]
    return df

def latest_trade_price(client, symbol: str) -> float:
    lt = client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
    return float(lt[symbol].price)

def fetch_news(symbol: str, limit: int = 5, include_content: bool = False):
    # News API via alpaca-py
    news_client = NewsClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    req = NewsRequest(
        symbols=[symbol],
        limit=limit,
        include_content=include_content,
        exclude_contentless=True
    )
    return news_client.get_news(req)

# ---------- GPT quick evaluation ----------
def build_eval_prompt(rep: dict) -> str:
    """
    Provide an unbiased stock analisys based on the data provided to provide extra insights to an investment group to better decide
    whether to invest in the stock or sell. Provide a comprehensive evaluation of risk, gain opportunities and an estimate
    of future performance in the short/long term. keep it under 300 words. Prefer schematic answers.
    """

    lines = []
    lines.append(f"Symbol: {rep['symbol']}")
    p = rep["price"]
    lines.append(f"Price Last: {p['last']} | Prev Close: {p['prev_close']} | 1D: {p['change_1d_pct']}%")

    lines.append("Returns (%):")
    for k, v in rep["returns_pct"].items():
        lines.append(f"  {k}: {v}")

    tr = rep["trend"]
    lines.append("Trend:")
    lines.append(f"  MA20={tr['MA20']}  MA50={tr['MA50']}  MA200={tr['MA200']}")
    lines.append(f"  52w High={tr['52w_high']} ({tr['from_52w_high_pct']}% from high)")
    lines.append(f"  52w Low={tr['52w_low']} ({tr['from_52w_low_pct']}% from low)")

    rk = rep["risk"]
    lines.append("Risk:")
    lines.append(f"  Vol(ann)={rk['realized_vol_annual']}  ATR14={rk['ATR14']} ({rk['ATR14_pct']}%)  RSI14={rk['RSI14']}")
    lines.append(f"  MaxDD={rk['max_drawdown_pct']}%  Sharpe20={rk['sharpe_20d']}  Sharpe60={rk['sharpe_60d']}  UpDays60={rk['pos_days_60d_pct']}%")

    liq = rep["liquidity"]
    lines.append("Liquidity:")
    lines.append(f"  ADV20={liq['ADV20']}  ADV50={liq['ADV50']}  ADV90={liq['ADV90']}  DollarVol20={liq['DollarVol20']}")

    rel = rep["relative"]
    lines.append("Relative vs SPY:")
    lines.append(f"  Beta={rel['beta_vs_SPY']}  Corr={rel['corr_vs_SPY']}  Alpha(ann%)={rel['alpha_annual_pct']}  TE(ann)={rel['tracking_error_annual']}")

    lines.append(
        "\nYou are a buy-side analyst. Provide a concise quick-read:\n"
        "- First line: BUY / HOLD / SELL with brief rationale.\n"
        "- Then 3–5 bullets: thesis angle, key risks, likely catalysts, and relative view vs SPY.\n"
        "- <= 120 words. No tables, no disclaimers."
    )
    return "\n".join(lines)

def gpt_quick_evaluation(rep: dict) -> str:
    if not ANALYZE_WITH_GPT:
        return ""
    if OpenAI is None or not OPENAI_API_KEY:
        return "(GPT evaluation unavailable: OpenAI client or API key not configured.)"
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = build_eval_prompt(rep)
        resp = client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {"role": "system", "content": "You are an expert equity analyst. Be concise and numerate."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=220,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(GPT evaluation error: {e})"

# ---------- Main Insight Builder ----------
def build_insights(symbol: str, period_str: str = "1y"):
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=parse_period_to_days(period_str))

    # Fetch bars
    bars = fetch_bars(client, symbol, start, end)
    if bars.empty or len(bars) < 30:
        raise RuntimeError(f"Not enough bar data for {symbol} in period {period_str}")

    # Also fetch benchmark bars for beta/correlation
    bench_bars = fetch_bars(client, BENCHMARK, start, end)
    # Align indices
    df = bars.copy()
    bench = bench_bars.copy()
    common = df.index.intersection(bench.index)
    df = df.loc[common]
    bench = bench.loc[common]

    # Latest price (prefer latest trade if available)
    try:
        last_price = latest_trade_price(client, symbol)
    except Exception:
        last_price = float(df['close'].iloc[-1])

    prev_close = float(df['close'].iloc[-2]) if len(df) >= 2 else np.nan
    change_1d = (last_price / prev_close - 1.0) * 100 if prev_close > 0 else np.nan

    # Returns
    daily_ret = df['close'].pct_change().dropna()
    bench_ret = bench['close'].pct_change().dropna()
    vol_annual = annualize_vol(daily_ret)

    returns = {}
    for w in RETURN_WINDOWS:
        if w.name == "YTD":
            returns[w.name] = ytd_change(df['close'])
        else:
            returns[w.name] = pct_change_over(df['close'], w.days)

    # Beta & correlation vs SPY
    join = pd.concat([daily_ret, bench_ret], axis=1, join="inner").dropna()
    join.columns = ['r_stock', 'r_bench']
    if len(join) >= 30:
        cov = float(np.cov(join['r_stock'], join['r_bench'])[0, 1])
        var_bench = float(np.var(join['r_bench']))
        beta = cov / var_bench if var_bench > 0 else np.nan
        corr = float(join['r_stock'].corr(join['r_bench']))
    else:
        beta, corr = np.nan, np.nan

    # Liquidity
    adv20 = float(df['volume'].tail(20).mean())
    adv50 = float(df['volume'].tail(50).mean()) if len(df) >= 50 else np.nan
    adv90 = float(df['volume'].tail(90).mean()) if len(df) >= 90 else np.nan
    dollar_vol20 = float((df['close'].tail(20) * df['volume'].tail(20)).mean())

    # Risk indicators
    rsi_14 = rsi(df['close'], 14)
    atr_14 = atr(df[['open','high','low','close']], 14)

    # Additional risk/quality stats
    atr_pct = float(atr_14 / last_price * 100) if last_price else np.nan
    mdd_pct = max_drawdown_pct(df['close'])
    sharpe_20 = rolling_sharpe(daily_ret, 20)
    sharpe_60 = rolling_sharpe(daily_ret, 60)
    up_days_60_pct = float((daily_ret.tail(60) > 0).mean() * 100) if len(daily_ret) >= 5 else np.nan

    # Relative performance vs benchmark
    if len(join) >= 30:
        alpha_annual_pct = float((join['r_stock'] - join['r_bench']).mean() * 252 * 100)
        tracking_error_ann = float((join['r_stock'] - join['r_bench']).std(ddof=1) * np.sqrt(252))
    else:
        alpha_annual_pct, tracking_error_ann = np.nan, np.nan

    # MAs
    ma20 = moving_average(df['close'], 20)
    ma50 = moving_average(df['close'], 50)
    ma200 = moving_average(df['close'], 200)

    # 52W range (use last 252 trading days if available)
    window = min(252, len(df))
    rolling = df['close'].tail(window)
    high_52w = float(rolling.max())
    low_52w = float(rolling.min())
    dist_from_high = (last_price / high_52w - 1.0) * 100 if high_52w else np.nan
    dist_from_low = (last_price / low_52w - 1.0) * 100 if low_52w else np.nan

    # News
    headlines = []
    try:
        news = fetch_news(symbol, NEWS_LIMIT)
        for item in news:
            headlines.append({
                "headline": item.headline,
                "source": item.source,
                "created_at": getattr(item, "created_at", None)
            })
    except Exception:
        pass

    # Save bars for inspection
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(f"outputs/{symbol}_bars.csv", index=True)

    # Build a printable report
    report = {
        "symbol": symbol,
        "as_of_utc": end.strftime("%Y-%m-%d %H:%M:%S"),
        "price": {
            "last": round(last_price, 4),
            "prev_close": round(prev_close, 4) if prev_close == prev_close else None,
            "change_1d_pct": None if np.isnan(change_1d) else round(change_1d, 2)
        },
        "returns_pct": {k: (None if v is None or np.isnan(v) else round(v, 2)) for k, v in returns.items()},
        "trend": {
            "MA20": None if np.isnan(ma20) else round(ma20, 4),
            "MA50": None if np.isnan(ma50) else round(ma50, 4),
            "MA200": None if np.isnan(ma200) else round(ma200, 4),
            "52w_high": round(high_52w, 4),
            "52w_low": round(low_52w, 4),
            "from_52w_high_pct": None if np.isnan(dist_from_high) else round(dist_from_high, 2),
            "from_52w_low_pct": None if np.isnan(dist_from_low) else round(dist_from_low, 2),
        },
        "risk": {
            "realized_vol_annual": round(vol_annual, 4),
            "ATR14": round(atr_14, 4),
            "ATR14_pct": None if np.isnan(atr_pct) else round(atr_pct, 2),
            "RSI14": round(rsi_14, 2),
            "max_drawdown_pct": None if np.isnan(mdd_pct) else round(mdd_pct, 2),
            "sharpe_20d": None if np.isnan(sharpe_20) else round(sharpe_20, 2),
            "sharpe_60d": None if np.isnan(sharpe_60) else round(sharpe_60, 2),
            "pos_days_60d_pct": None if np.isnan(up_days_60_pct) else round(up_days_60_pct, 1),
        },
        "liquidity": {
            "ADV20": int(adv20),
            "ADV50": None if np.isnan(adv50) else int(adv50),
            "ADV90": None if np.isnan(adv90) else int(adv90),
            "DollarVol20": int(dollar_vol20),
        },
        "relative": {
            "beta_vs_SPY": None if np.isnan(beta) else round(beta, 3),
            "corr_vs_SPY": None if np.isnan(corr) else round(corr, 3),
            "alpha_annual_pct": None if np.isnan(alpha_annual_pct) else round(alpha_annual_pct, 2),
            "tracking_error_annual": None if np.isnan(tracking_error_ann) else round(tracking_error_ann, 4),
        },
        "news": headlines
    }

    return report

def pretty_print_report(rep: dict):
    print(f"\n=== Insights for {rep['symbol']} (as of {rep['as_of_utc']} UTC) ===")
    p = rep["price"]
    print(f"Price: {p['last']}  | Prev Close: {p['prev_close']}  | 1D: {p['change_1d_pct']}%")
    print("\nReturns:")
    for k, v in rep["returns_pct"].items():
        print(f"  {k:>3}: {v}%")
    tr = rep["trend"]
    print("\nTrend:")
    print(f"  MA20: {tr['MA20']}  | MA50: {tr['MA50']}  | MA200: {tr['MA200']}")
    print(f"  52w High: {tr['52w_high']} ({tr['from_52w_high_pct']}% from high)")
    print(f"  52w Low : {tr['52w_low']} ({tr['from_52w_low_pct']}% from low)")
    rk = rep["risk"]
    print("\nRisk:")
    print(f"  Realized Vol (ann): {rk['realized_vol_annual']}")
    print(f"  ATR-14: {rk['ATR14']}   | RSI-14: {rk['RSI14']}")
    print(f"  ATR-14 %: {rk['ATR14_pct']}   | Max DD: {rk['max_drawdown_pct']}%")
    print(f"  Sharpe20: {rk['sharpe_20d']} | Sharpe60: {rk['sharpe_60d']} | Up-days(60d): {rk['pos_days_60d_pct']}%")
    liq = rep["liquidity"]
    print("\nLiquidity:")
    print(f"  ADV20: {liq['ADV20']}  | ADV50: {liq['ADV50']}  | ADV90: {liq['ADV90']}")
    print(f"  DollarVol20 (avg): {liq['DollarVol20']}")
    rel = rep["relative"]
    print("\nRelative (vs SPY):")
    print(f"  Beta: {rel['beta_vs_SPY']}  | Corr: {rel['corr_vs_SPY']}")
    print(f"  Alpha (annual, %): {rel['alpha_annual_pct']}  | Tracking Error (ann): {rel['tracking_error_annual']}")
    if rep["news"]:
        print("\nLatest Headlines:")
        for n in rep["news"]:
            ts = getattr(n["created_at"], "strftime", lambda *_: None)("%Y-%m-%d %H:%M") if n["created_at"] else ""
            print(f"  - [{ts}] {n['source']}: {n['headline']}")
    # Optional GPT quick evaluation
    if ANALYZE_WITH_GPT:
        print("\n[AI] Recap:")
        print(gpt_quick_evaluation(rep))

    print("-------------------------------------------------------------\n")

def main():
    rep = build_insights(SYMBOL, PERIOD)
    pretty_print_report(rep)

if __name__ == "__main__":
    main()
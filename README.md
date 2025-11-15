#  Stock Analysis Bot

This repository contains a Python script that builds a detailed, data-driven snapshot of a stock, optionally enriched with a short AI-generated recap.

The main file is:

- `stock_investment_gpt.py` – fetches historical market data, computes risk/return metrics, and (optionally) sends the results to OpenAI for a quick textual evaluation.

---

## What the script does

For a given stock symbol, the script:

- Fetches **daily price data** and **volume** from Alpaca’s market data API.
- Computes:
  - Returns over several windows (1W, 1M, 3M, 6M, 1Y, YTD)
  - Realized annual volatility
  - RSI(14), ATR(14), max drawdown
  - Moving averages (20d, 50d, 200d)
  - 52-week high/low and distance from them
  - Liquidity stats (ADV20/50/90, dollar volume)
  - Relative stats vs **SPY**: beta, correlation, alpha, tracking error
- Optionally calls **OpenAI** to generate a short “BUY / HOLD / SELL” style recap based on the computed metrics.
- Prints a human-readable report to the console and saves the raw bar data to `outputs/<SYMBOL>_bars.csv`.

---

## Data sources

- **Price & volume data:**  
  Via `alpaca-py` `StockHistoricalDataClient`  
  (using the `iex` feed by default, suitable for US stocks).

- **News (optional):**  
  Via `alpaca-py` `NewsClient` for recent headlines on the selected symbol.

- **AI recap (optional):**  
  Uses the OpenAI Chat Completions API (model configurable via `.env`).

---

## Requirements

- Python 3.10+ (recommended)
- Dependencies listed in `requirements.txt`, typically including:
  - `alpaca-py`
  - `pandas`
  - `numpy`
  - `python-dotenv`
  - `openai` (for GPT analysis)

Install them with:

```bash
pip install -r requirements.txt

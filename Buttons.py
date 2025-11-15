import os
import logging
from dotenv import load_dotenv

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# Import your logic from the existing script
from stock_investment_gpt import (
    build_insights,
    gpt_quick_evaluation,
    ANALYZE_WITH_GPT,
)

# ---------- ENV & CONFIG ----------

load_dotenv()  # reuse same .env file

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEFAULT_PERIOD = os.getenv("PERIOD", "1y")

# Only the periods you asked for (and that we’ll show as buttons)
ALLOWED_PERIODS = ["3m", "6m", "1y"]

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Please set TELEGRAM_BOT_TOKEN in your .env file")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------- HELPERS ----------

def clean_ticker(text: str) -> str:
    """
    Turn a raw message into a candidate ticker (e.g. 'AAPL ' -> 'AAPL').
    Very simple: keep only letters/digits and uppercase.
    """
    filtered = "".join(ch for ch in text.strip() if ch.isalnum())
    return filtered.upper()


def get_user_period(context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    Get the preferred period for this user/chat, falling back to DEFAULT_PERIOD.
    """
    return context.user_data.get("period", DEFAULT_PERIOD)


def period_keyboard(selected: str | None = None) -> InlineKeyboardMarkup:
    """
    Inline keyboard with the 4 timeframes.
    The currently selected one is wrapped in brackets: [6m]
    """
    buttons = []
    row = []
    for p in ALLOWED_PERIODS:
        label = f"[{p}]" if p == (selected or "").lower() else p
        row.append(
            InlineKeyboardButton(
                text=label,
                callback_data=f"PERIOD:{p}",
            )
        )
    buttons.append(row)
    return InlineKeyboardMarkup(buttons)


def format_report_for_telegram(rep: dict) -> str:
    """
    Turn the report dict from build_insights() into a compact text
    suitable for Telegram messages.
    """
    p = rep["price"]
    tr = rep["trend"]
    rk = rep["risk"]
    liq = rep["liquidity"]
    rel = rep["relative"]

    lines = []
    lines.append(f"📊 *{rep['symbol']}*  (as of {rep['as_of_utc']} UTC)")
    lines.append(
        f"Price: *{p['last']}*  | Prev Close: {p['prev_close']}  "
        f"| 1D: {p['change_1d_pct']}%"
    )

    # Returns (short)
    lines.append("\n*Returns (%):*")
    for k, v in rep["returns_pct"].items():
        lines.append(f"• {k}: {v}%")

    # Trend
    lines.append("\n*Trend:*")
    lines.append(
        f"• MA20: {tr['MA20']} | MA50: {tr['MA50']} | MA200: {tr['MA200']}"
    )
    lines.append(
        f"• 52w High: {tr['52w_high']} ({tr['from_52w_high_pct']}% from high)"
    )
    lines.append(
        f"• 52w Low : {tr['52w_low']} ({tr['from_52w_low_pct']}% from low)"
    )

    # Risk
    lines.append("\n*Risk:*")
    lines.append(f"• Vol (ann): {rk['realized_vol_annual']}")
    lines.append(f"• ATR14: {rk['ATR14']} ({rk['ATR14_pct']}%)")
    lines.append(f"• RSI14: {rk['RSI14']}")
    lines.append(f"• Max DD: {rk['max_drawdown_pct']}%")
    lines.append(
        f"• Sharpe20: {rk['sharpe_20d']} | Sharpe60: {rk['sharpe_60d']} "
        f"| Up-days (60d): {rk['pos_days_60d_pct']}%"
    )

    # Liquidity
    lines.append("\n*Liquidity:*")
    lines.append(
        f"• ADV20: {liq['ADV20']} | ADV50: {liq['ADV50']} | ADV90: {liq['ADV90']}"
    )
    lines.append(f"• $Vol20 (avg): {liq['DollarVol20']}")

    # Relative
    lines.append("\n*Relative vs SPY:*")
    lines.append(
        f"• Beta: {rel['beta_vs_SPY']} | Corr: {rel['corr_vs_SPY']}"
    )
    lines.append(
        f"• Alpha (ann, %): {rel['alpha_annual_pct']} | TE (ann): {rel['tracking_error_annual']}"
    )

    # Optional headlines
    if rep["news"]:
        lines.append("\n*Latest Headlines:*")
        for n in rep["news"]:
            ts = (
                n["created_at"].strftime("%Y-%m-%d")
                if n["created_at"] is not None
                else ""
            )
            lines.append(f"• [{ts}] {n['source']}: {n['headline']}")

    return "\n".join(lines)


async def send_analysis(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    symbol: str,
    period: str,
) -> None:
    """
    Shared function to run analysis and send messages.
    Works for both normal messages and callback queries.
    """
    chat_id = update.effective_chat.id

    await context.bot.send_message(
        chat_id,
        f"🔎 Analyzing *{symbol}* with period `{period}` ...",
        parse_mode="Markdown",
    )

    try:
        rep = build_insights(symbol, period)
        text_report = format_report_for_telegram(rep)

        # Main report + timeframe buttons
        await context.bot.send_message(
            chat_id,
            text_report,
            parse_mode="Markdown",
            reply_markup=period_keyboard(selected=period),
        )

        # Optional GPT view
        if ANALYZE_WITH_GPT:
            gpt_view = gpt_quick_evaluation(rep)
            if gpt_view:
                await context.bot.send_message(
                    chat_id,
                    f"🤖 *AI view on {symbol}:*\n{gpt_view}",
                    parse_mode="Markdown",
                )

    except Exception as e:
        logger.exception("Error in send_analysis", exc_info=e)
        await context.bot.send_message(
            chat_id,
            f"⚠️ Sorry, I couldn't analyze `{symbol}`.\nError: `{e}`",
            parse_mode="Markdown",
        )


# ---------- HANDLERS ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    period = get_user_period(context)
    msg = (
        "Welcome to the *Medici* Stock Analysis Bot 🦁 \n\n"
        "• Please indicate a *stock ticker* (i.e. AAPL) and I'll provide a quantitative snapshot.\n\n"
         f"• Current default lookback period: `{period}`\n\n"
        "You can also make a request for a *specific period* (i.e. AAPL 6m).\n\n"
        "Or use the buttons below after an analysis to quickly switch between:\n"
        "`3m`, `6m` or `1y`."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "Usage:\n"
        "• Just send a ticker symbol, e.g. `AAPL`\n"
        "• Or send `AAPL 6m` / `MSFT 1y` to specify the lookback.\n\n"
        "I’ll fetch OHLCV data, compute trends, risk stats, liquidity, and relative performance.\n"
        "If GPT analysis is enabled in admin's code, you'll also get a short AI view.\n\n"
        "After I send a report, you can tap one of the timeframe buttons:\n"
        "`3m`, `6m`, `1y` to rerun the analysis for the same ticker.\n"
        "The one in brackets is the currently selected."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def handle_ticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle any non-command text as a ticker request.
    Supports optional timeframe: 'AAPL 3m', 'AAPL 6m', or 'AAPL 1y'.
    """
    if not update.message or not update.message.text:
        return

    raw_text = update.message.text.strip()

    # Default: use the user's saved period (or global DEFAULT_PERIOD)
    period = get_user_period(context)

    # If the user sends something like "AAPL 6m", treat the last token as potential period
    parts = raw_text.split()
    if len(parts) >= 2 and parts[-1].lower() in ALLOWED_PERIODS:
        period = parts[-1].lower()
        symbol_text = " ".join(parts[:-1])
    else:
        symbol_text = raw_text

    symbol = clean_ticker(symbol_text)

    if not symbol:
        await update.message.reply_text(
            "Please send a valid stock ticker symbol, e.g. `AAPL` or `MSFT`.",
            parse_mode="Markdown",
        )
        return

    # Save latest ticker & period for this chat/user so buttons can reuse them
    context.user_data["last_symbol"] = symbol
    context.user_data["period"] = period

    await send_analysis(update, context, symbol, period)


async def period_button_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Handle button presses like 'PERIOD:6m'.
    Reuses the last symbol from user_data.
    """
    query = update.callback_query
    await query.answer()

    data = (query.data or "").strip()
    if not data.startswith("PERIOD:"):
        return

    period = data.split(":", 1)[1].lower()
    if period not in ALLOWED_PERIODS:
        await query.message.reply_text(
            "❌ Invalid period received. Please try again.",
            parse_mode="Markdown",
        )
        return

    symbol = context.user_data.get("last_symbol")
    if not symbol:
        await query.message.reply_text(
            "Please send me a ticker first, e.g. `AAPL`.",
            parse_mode="Markdown",
        )
        return

    # Update stored period and rerun analysis
    context.user_data["period"] = period
    await send_analysis(update, context, symbol, period)


# ---------- MAIN ----------

def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))

    # Inline button callbacks for periods
    app.add_handler(CallbackQueryHandler(period_button_handler, pattern=r"^PERIOD:"))

    # Any non-command text is treated as a ticker
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ticker))

    app.run_polling()


if __name__ == "__main__":
    main()
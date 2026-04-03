"""
main.py -- Loop principal do Boris v3
======================================
- Scheduler para análises periódicas
- Comandos: /start, /analise, /preco, /historico, /config
- Envio de gráfico de preço
- Passa price_trend_pct corretamente para run_analysis
"""

import argparse
import asyncio
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("boris.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("boris.main")

from tools import (
    fetch_bitcoin_price,
    fetch_bitcoin_news,
    fetch_bitcoin_ohlc,
    compute_indicators,
    generate_price_chart,
    format_price_summary,
)
from memory import AgentMemory
from analysis import run_analysis, AnalysisResult

# ---------------------------------------------------------------------------
# Configurações via .env
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CHECK_INTERVAL_HOURS = float(os.getenv("CHECK_INTERVAL_HOURS", "3"))
ALERT_THRESHOLD_PCT = float(os.getenv("ALERT_THRESHOLD_PERCENT", "5.0"))
LLM_MODEL = os.getenv("LLM_MODEL", "microsoft/Phi-3-mini-4k-instruct")
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "400"))

CHECK_INTERVAL_SECONDS = CHECK_INTERVAL_HOURS * 3600
MAX_PRICE_HISTORY = 24

TREND_LABEL = {
    "Alta":   "🟢 ALTA",
    "Baixa":  "🔴 BAIXA",
    "Neutra": "⚪ NEUTRA",
}

memory = AgentMemory()
application: Application = None  # inicializado em main()


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _extract_trend_pct(summary: str) -> float:
    """Extrai o percentual numérico do resumo de tendência da memória."""
    m = re.search(r"(subiu|caiu)\s+([\d,\.]+)%", summary)
    if not m:
        return 0.0
    pct = float(m.group(2).replace(",", "."))
    return pct if m.group(1) == "subiu" else -pct


# ---------------------------------------------------------------------------
# Ciclo completo de análise
# ---------------------------------------------------------------------------

async def perform_analysis(chat_id: str = None, is_alert: bool = False) -> AnalysisResult:
    """Coleta dados, roda análise e envia relatório pelo Telegram."""

    # 1. Preço atual
    price_data = fetch_bitcoin_price()
    if not price_data:
        logger.error("Falha ao obter preço. Análise abortada.")
        return None

    # 2. Notícias
    news_list = fetch_bitcoin_news(max_news=3)

    # 3. Persistência na memória
    memory.add_price(price_data)
    if news_list:
        memory.add_news(news_list)

    # 4. Indicadores técnicos
    ohlc = fetch_bitcoin_ohlc(days=7)
    indicators = compute_indicators(ohlc) if (ohlc is not None and not ohlc.empty) else {}

    # 5. Resumos da memória
    price_trend_summary = memory.get_price_trend_summary()
    decision_summary = memory.get_decision_summary()

    # 6. Percentual de tendência histórica (para o score composto)
    price_trend_pct = _extract_trend_pct(price_trend_summary)

    # 7. Análise
    result = run_analysis(
        price_data=price_data,
        news_list=news_list,
        price_trend_summary=price_trend_summary,
        decision_summary=decision_summary,
        model_name=LLM_MODEL,
        indicators=indicators,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        price_trend_pct=price_trend_pct,
    )

    # 8. Salvar decisão
    memory.add_decision(result.trend, result.recommendation)

    # 9. Relatório e gráfico
    report = _build_report(price_data, news_list, result, indicators, is_alert)
    price_history = memory.get_recent_prices(MAX_PRICE_HISTORY)
    chart = generate_price_chart(price_history) if price_history else None

    # 10. Envio
    target = chat_id or TELEGRAM_CHAT_ID
    if target:
        await _send_message(target, report, photo=chart)

    return result


async def _send_message(chat_id: str, text: str, photo: BytesIO = None):
    if not application:
        logger.error("Bot não inicializado.")
        return
    try:
        if photo:
            await application.bot.send_photo(
                chat_id=chat_id, photo=photo, caption=text, parse_mode="HTML"
            )
        else:
            await application.bot.send_message(
                chat_id=chat_id, text=text, parse_mode="HTML"
            )
    except Exception as e:
        logger.error("Erro ao enviar mensagem: %s", e)


def _build_report(price_data: dict, news_list: list[dict],
                   result: AnalysisResult, indicators: dict, is_alert: bool) -> str:
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    trend_label = TREND_LABEL.get(result.trend, result.trend)
    mode_label = "LLM" if result.used_llm else "Técnica"
    sign = "+" if price_data["change_24h"] >= 0 else ""

    # Indicadores
    ind_parts = []
    rsi = indicators.get("rsi")
    if rsi is not None:
        state = "sobrecompra" if rsi > 70 else "sobrevenda" if rsi < 30 else "neutro"
        ind_parts.append(f"RSI(14): {rsi:.1f} ({state})")
    sma7 = indicators.get("sma7")
    sma25 = indicators.get("sma25")
    if sma7 and sma25:
        cross = "acima" if sma7 > sma25 else "abaixo"
        ind_parts.append(f"MM7 {cross} da MM25")
    ind_text = " | ".join(ind_parts) if ind_parts else "Indisponível"

    # Score interno (debug leve)
    score_text = f"Score: {result.score:+.3f}"

    alert_header = "<b>🚨 ALERTA — VARIAÇÃO SIGNIFICATIVA</b>\n\n" if is_alert else ""

    news_lines = []
    for i, n in enumerate(news_list[:3], 1):
        title = n["title"][:90] + ("..." if len(n["title"]) > 90 else "")
        sent = n.get("sentiment", 0.0)
        emoji = "🟢" if sent > 0.2 else "🔴" if sent < -0.2 else "⚪"
        news_lines.append(f"  {i}. {emoji} {title}")
    news_block = "\n".join(news_lines) if news_lines else "  Nenhuma notícia."

    report = (
        f"{alert_header}"
        f"<b>BORIS — Relatório Bitcoin</b>\n"
        f"<i>{now}</i>\n\n"
        f"<b>Preço:</b> <code>${price_data['price_usd']:,.2f}</code> USD\n"
        f"<b>Variação 24h:</b> <code>{sign}{price_data['change_24h']:.2f}%</code>\n"
        f"<b>Volume 24h:</b> <code>${price_data['volume_24h']:,.0f}</code>\n\n"
        f"<b>Indicadores:</b> {ind_text}\n"
        f"<b>{score_text}</b>\n\n"
        f"<b>Notícias:</b>\n{news_block}\n\n"
        f"<b>Tendência:</b> {trend_label}\n"
        f"<b>Confiança:</b> {result.confidence}\n\n"
        f"<b>Recomendação:</b>\n<i>{result.recommendation}</i>\n\n"
        f"<i>Análise via {mode_label} | Boris v3.0</i>\n"
        f"<i>Programado por <b>Eduardo Araujo (@lalo_araujo)</b> 😊</i>"
    )
    return report


# ---------------------------------------------------------------------------
# Comandos do Telegram
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Olá! Sou o Boris v3, seu estrategista de Bitcoin.\n\n"
        "Comandos disponíveis:\n"
        "/analise — Análise completa agora\n"
        "/preco   — Preço atual + indicadores\n"
        "/historico — Últimas decisões\n"
        "/config  — Configurações atuais\n\n"
        "Análises periódicas são enviadas automaticamente a cada "
        f"{CHECK_INTERVAL_HOURS}h."
    )


async def cmd_analise(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Coletando dados e gerando análise...")
    result = await perform_analysis(chat_id=str(update.effective_chat.id))
    if result is None:
        await update.message.reply_text("❌ Falha ao obter dados. Tente novamente.")


async def cmd_preco(update: Update, context: ContextTypes.DEFAULT_TYPE):
    price = fetch_bitcoin_price()
    if not price:
        await update.message.reply_text("❌ Erro ao obter preço.")
        return
    ohlc = fetch_bitcoin_ohlc(days=7)
    indicators = compute_indicators(ohlc) if (ohlc is not None and not ohlc.empty) else {}
    text = format_price_summary(price)
    rsi = indicators.get("rsi")
    if rsi is not None:
        state = "sobrecompra" if rsi > 70 else "sobrevenda" if rsi < 30 else "neutro"
        text += f"\nRSI(14): {rsi:.1f} ({state})"
    sma7 = indicators.get("sma7")
    sma25 = indicators.get("sma25")
    if sma7 and sma25:
        cross = "acima" if sma7 > sma25 else "abaixo"
        text += f"\nMM7 {cross} da MM25 (${sma7:,.0f} / ${sma25:,.0f})"
    await update.message.reply_text(text)


async def cmd_historico(update: Update, context: ContextTypes.DEFAULT_TYPE):
    summary = memory.get_decision_summary()
    await update.message.reply_text(summary)


async def cmd_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        f"⚙️ Configurações do Boris v3\n"
        f"Intervalo de análise: {CHECK_INTERVAL_HOURS}h\n"
        f"Limiar de alerta: {ALERT_THRESHOLD_PCT}%\n"
        f"Modelo LLM: {LLM_MODEL}\n"
        f"Memória: {memory.stats()}"
    )
    await update.message.reply_text(text)


# ---------------------------------------------------------------------------
# Scheduler de análises automáticas
# ---------------------------------------------------------------------------

def _scheduler_loop():
    while True:
        time.sleep(CHECK_INTERVAL_SECONDS)
        try:
            asyncio.run(perform_analysis(chat_id=TELEGRAM_CHAT_ID))
        except Exception as e:
            logger.exception("Erro na análise agendada: %s", e)


def start_scheduler():
    thread = threading.Thread(target=_scheduler_loop, daemon=True)
    thread.start()
    logger.info("Scheduler iniciado (intervalo %.0fs)", CHECK_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Boris — Bot de análise de Bitcoin")
    parser.add_argument("--once", action="store_true", help="Executa uma análise e encerra")
    parser.add_argument("--test", action="store_true", help="Testa conexões")
    args = parser.parse_args()

    if args.test:
        _run_test()
        return

    global application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("analise", cmd_analise))
    application.add_handler(CommandHandler("preco", cmd_preco))
    application.add_handler(CommandHandler("historico", cmd_historico))
    application.add_handler(CommandHandler("config", cmd_config))

    start_scheduler()

    if args.once:
        asyncio.run(perform_analysis(chat_id=TELEGRAM_CHAT_ID))
        return

    logger.info("Boris v3 iniciado. Aguardando comandos...")
    application.run_polling()


def _run_test():
    from tools import format_price_summary, fetch_bitcoin_news, fetch_bitcoin_ohlc, compute_indicators
    logger.info("=== TESTE DE CONEXÕES ===")
    price = fetch_bitcoin_price()
    if price:
        logger.info("✅ Preço OK: %s", format_price_summary(price))
    else:
        logger.error("❌ Falha ao obter preço.")
    news = fetch_bitcoin_news(3)
    logger.info("✅ Notícias: %d obtidas", len(news))
    ohlc = fetch_bitcoin_ohlc(7)
    if ohlc is not None:
        ind = compute_indicators(ohlc)
        logger.info("✅ OHLC OK | RSI=%.1f | MM7=%.0f | MM25=%.0f",
                    ind.get("rsi", 0), ind.get("sma7", 0), ind.get("sma25", 0))
    logger.info("✅ Memória: %s", memory.stats())
    logger.info("=== TESTE CONCLUÍDO ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Boris encerrado pelo usuário.")
    except Exception as e:
        logger.exception("Erro crítico: %s", e)

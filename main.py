"""
main.py -- Loop principal com comandos Telegram e gráficos
===========================================================
- Scheduler para análises periódicas
- Comandos: /analise, /preco, /historico, /config
- Envio de gráfico de preço
"""

import argparse
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("boris.log", encoding="utf-8")],
)
logger = logging.getLogger("boris.main")

from tools import (
    fetch_bitcoin_price, fetch_bitcoin_news, fetch_bitcoin_ohlc, compute_indicators,
    generate_price_chart, format_price_summary, format_news_summary
)
from memory import AgentMemory
from analysis import run_analysis, AnalysisResult

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # pode ser usado para envio automático
CHECK_INTERVAL_HOURS = float(os.getenv("CHECK_INTERVAL_HOURS", "3"))
ALERT_THRESHOLD_PCT = float(os.getenv("ALERT_THRESHOLD_PERCENT", "5.0"))
LLM_MODEL = os.getenv("LLM_MODEL", "microsoft/Phi-3-mini-4k-instruct")
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "400"))

CHECK_INTERVAL_SECONDS = CHECK_INTERVAL_HOURS * 3600
TREND_LABEL = {"Alta": "🟢 ALTA", "Baixa": "🔴 BAIXA", "Neutra": "⚪ NEUTRA"}
CONF_LABEL = {"Alta": "Alta", "Media": "Média", "Baixa": "Baixa"}
MAX_PRICE_HISTORY = 24

memory = AgentMemory()
application = None  # será inicializado depois

# ----------------------------------------------
# Funções auxiliares de análise e envio
# ----------------------------------------------
async def perform_analysis(chat_id: str = None, is_alert: bool = False) -> AnalysisResult:
    """Executa um ciclo completo de análise e retorna o resultado."""
    price_data = fetch_bitcoin_price()
    if not price_data:
        return None
    news_list = fetch_bitcoin_news(max_news=3)
    memory.add_price(price_data)
    if news_list:
        memory.add_news(news_list)

    # Obter indicadores técnicos (OHLC)
    ohlc = fetch_bitcoin_ohlc(days=7)
    indicators = compute_indicators(ohlc) if ohlc is not None else {}

    price_trend_summary = memory.get_price_trend_summary()
    decision_summary = memory.get_decision_summary()

    result = run_analysis(
        price_data, news_list, price_trend_summary, decision_summary,
        LLM_MODEL, indicators, LLM_MAX_NEW_TOKENS
    )
    memory.add_decision(result.trend, result.recommendation)

    # Montar relatório
    report = build_report(price_data, news_list, result, indicators, is_alert)
    # Gerar gráfico
    price_history = memory.get_recent_prices(MAX_PRICE_HISTORY)  # precisamos definir MAX_PRICE_HISTORY
    chart = generate_price_chart(price_history) if price_history else None

    # Enviar via bot (se chat_id fornecido)
    if chat_id:
        await send_telegram_message(chat_id, report, photo=chart)
    else:
        # Fallback para o chat configurado
        if TELEGRAM_CHAT_ID:
            await send_telegram_message(TELEGRAM_CHAT_ID, report, photo=chart)
    return result

async def send_telegram_message(chat_id: str, text: str, photo: BytesIO = None):
    """Envia mensagem com ou sem foto."""
    if not application:
        logger.error("Bot não inicializado.")
        return
    if photo:
        await application.bot.send_photo(chat_id=chat_id, photo=photo, caption=text, parse_mode='HTML')
    else:
        await application.bot.send_message(chat_id=chat_id, text=text, parse_mode='HTML')

def build_report(price_data: dict, news_list: list[dict], result: AnalysisResult,
                 indicators: dict, is_alert: bool) -> str:
    now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    trend_label = TREND_LABEL.get(result.trend, result.trend)
    mode_label = "LLM" if result.used_llm else "Regras"
    sign = "+" if price_data["change_24h"] >= 0 else ""

    # Indicadores
    ind_text = ""
    if indicators:
        rsi = indicators.get('rsi')
        if rsi is not None:
            ind_text += f"RSI(14): {rsi:.2f} "
        sma7 = indicators.get('sma7')
        if sma7 is not None:
            ind_text += f"MM7: ${sma7:,.0f} "
        sma25 = indicators.get('sma25')
        if sma25 is not None:
            ind_text += f"MM25: ${sma25:,.0f}"

    alert_header = "<b>🚨 ALERTA - VARIAÇÃO SIGNIFICATIVA</b>\n\n" if is_alert else ""

    news_lines = []
    for i, n in enumerate(news_list[:3], 1):
        title = n["title"][:90] + ("..." if len(n["title"]) > 90 else "")
        news_lines.append(f"  {i}. {title}")
    news_block = "\n".join(news_lines) if news_lines else "  Nenhuma notícia."

    report = (
        f"{alert_header}"
        f"<b>BORIS -- Relatório Bitcoin</b>\n"
        f"<i>{now}</i>\n\n"
        f"<b>Preço:</b> <code>${price_data['price_usd']:,.2f}</code> USD\n"
        f"<b>Variação 24h:</b> <code>{sign}{price_data['change_24h']:.2f}%</code>\n"
        f"<b>Volume 24h:</b> <code>${price_data['volume_24h']:,.0f}</code>\n\n"
        f"<b>Indicadores:</b> {ind_text}\n\n"
        f"<b>Notícias:</b>\n{news_block}\n\n"
        f"<b>Tendência:</b> {trend_label}\n"
        f"<b>Confiança:</b> {result.confidence}\n\n"
        f"<b>Recomendação:</b>\n<i>{result.recommendation}</i>\n\n"
        f"<i>Análise via {mode_label} | Boris v2.0</i>\n"
        f"<i>Eu sou o Bóris — o bot de análise de Bitcoin programado por "
        f"<b>Eduardo Araujo (@lalo_araujo)</b> — meu painho. 😊</i>"
    )
    return report

# ----------------------------------------------
# Comandos do Telegram
# ----------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Olá! Sou o Boris, seu estrategista de Bitcoin.\n"
        "Comandos disponíveis:\n"
        "/analise - Executa uma análise agora\n"
        "/preco - Mostra preço atual e indicadores\n"
        "/historico - Últimas decisões\n"
        "/config - Mostra configurações atuais\n"
        "As análises periódicas são enviadas automaticamente."
    )

async def analise(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Coletando dados e gerando análise...")
    result = await perform_analysis(chat_id=str(update.effective_chat.id))
    if result is None:
        await update.message.reply_text("❌ Falha ao obter dados. Tente novamente mais tarde.")

async def preco(update: Update, context: ContextTypes.DEFAULT_TYPE):
    price = fetch_bitcoin_price()
    if not price:
        await update.message.reply_text("Erro ao obter preço.")
        return
    ohlc = fetch_bitcoin_ohlc(days=7)
    indicators = compute_indicators(ohlc) if (ohlc is not None and not ohlc.empty) else {}
    text = format_price_summary(price)
    if indicators.get('rsi'):
        text += f"\nRSI(14): {indicators['rsi']:.2f}"
    if indicators.get('sma7'):
        text += f"\nMM7: ${indicators['sma7']:,.0f}"
    if indicators.get('sma25'):
        text += f"\nMM25: ${indicators['sma25']:,.0f}"
    await update.message.reply_text(text)

async def historico(update: Update, context: ContextTypes.DEFAULT_TYPE):
    summary = memory.get_decision_summary()
    await update.message.reply_text(summary)

async def config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        f"Configurações atuais:\n"
        f"Intervalo: {CHECK_INTERVAL_HOURS}h\n"
        f"Alerta: {ALERT_THRESHOLD_PCT}%\n"
        f"Modelo: {LLM_MODEL}\n"
        f"Memória: {memory.stats()}"
    )
    await update.message.reply_text(text)

# ----------------------------------------------
# Scheduler para análises automáticas
# ----------------------------------------------
def schedule_regular_analysis():
    """Executa em thread separada, agendando análises a cada intervalo."""
    while True:
        now = time.time()
        next_run = now + CHECK_INTERVAL_SECONDS
        time.sleep(CHECK_INTERVAL_SECONDS)
        # Executar análise em modo background (não interativo)
        try:
            # Usar asyncio.run para chamar a função assíncrona
            import asyncio
            asyncio.run(perform_analysis(chat_id=TELEGRAM_CHAT_ID))
        except Exception as e:
            logger.exception("Erro na análise agendada: %s", e)

def start_scheduler():
    thread = threading.Thread(target=schedule_regular_analysis, daemon=True)
    thread.start()
    logger.info("Scheduler iniciado (intervalo %ds)", CHECK_INTERVAL_SECONDS)

# ----------------------------------------------
# Ponto de entrada
# ----------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Executa apenas um ciclo e encerra")
    parser.add_argument("--test", action="store_true", help="Testa conexões e exibe relatório")
    args = parser.parse_args()

    if args.test:
        run_test()
        return

    global application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("analise", analise))
    application.add_handler(CommandHandler("preco", preco))
    application.add_handler(CommandHandler("historico", historico))
    application.add_handler(CommandHandler("config", config))

    # Iniciar scheduler em thread separada
    start_scheduler()

    if args.once:
        # Executa uma análise manual e encerra
        import asyncio
        asyncio.run(perform_analysis(chat_id=TELEGRAM_CHAT_ID))
        return

    # Iniciar polling do bot
    logger.info("Boris iniciado, aguardando comandos...")
    application.run_polling()

def run_test():
    logger.info("=== TESTE ===")
    price = fetch_bitcoin_price()
    if price:
        logger.info("Preço OK: %s", format_price_summary(price))
    news = fetch_bitcoin_news(3)
    if news:
        logger.info("Notícias OK: %d", len(news))
    ohlc = fetch_bitcoin_ohlc(7)
    if ohlc is not None:
        logger.info("OHLC OK: %d registros", len(ohlc))
    memory = AgentMemory()
    logger.info("Memória: %s", memory.stats())
    logger.info("Teste concluído.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Boris encerrado.")
    except Exception as e:
        logger.exception("Erro crítico: %s", e)

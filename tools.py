"""
tools.py -- Ferramentas de coleta de dados do Boris
====================================================
Adicionado:
- Busca de histórico OHLC (últimos 7 dias) via CoinGecko
- Cálculo de RSI e médias móveis
- Análise de sentimento com modelo small (nlptown/bert-base-multilingual-uncased-sentiment)
- Retry com backoff exponencial via tenacity
- Função para gerar gráfico do preço com matplotlib
"""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd
import ta  # technical indicators
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO

logger = logging.getLogger(__name__)

# Constantes
COINGECKO_PRICE_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=bitcoin&vs_currencies=usd"
    "&include_24hr_change=true"
    "&include_24hr_vol=true"
    "&include_market_cap=true"
)

COINGECKO_OHLC_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days="

RSS_FEEDS = [
    "https://livecoins.com.br/feed/",
    "https://cryptoid.com.br/feed/",
    "https://br.cointelegraph.com/rss",
    "https://portaldobitcoin.uol.com.br/feed/",
    "https://www.infomoney.com.br/feed/",
    "https://valor.globo.com/rss/valor/",
    "https://investnews.com.br/feed/"
]

HEADERS = {"User-Agent": "BorisBot/2.0"}
REQUEST_TIMEOUT = 15

# Sentiment model (carregado lazy)
_sentiment_pipeline = None

# ----------------------------------------------
# Preço e dados históricos
# ----------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
       retry=retry_if_exception_type((requests.RequestException, ConnectionError)))
def fetch_bitcoin_price() -> Optional[dict]:
    """Busca preço atual e dados de mercado (com retry)."""
    try:
        response = requests.get(COINGECKO_PRICE_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        btc = data.get("bitcoin", {})
        if not btc:
            logger.error("CoinGecko não retornou dados do Bitcoin.")
            return None
        result = {
            "price_usd":  btc.get("usd", 0.0),
            "change_24h": btc.get("usd_24h_change", 0.0),
            "volume_24h": btc.get("usd_24h_vol", 0.0),
            "market_cap": btc.get("usd_market_cap", 0.0),
            "timestamp":  datetime.utcnow().isoformat() + "Z",
        }
        logger.info("Preço coletado: $%.2f (%.2f%%)", result["price_usd"], result["change_24h"])
        return result
    except Exception as e:
        logger.exception("Erro ao buscar preço: %s", e)
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_bitcoin_ohlc(days: int = 7) -> Optional[pd.DataFrame]:
    """
    Busca dados OHLC (Open, High, Low, Close) dos últimos `days` dias.
    Retorna DataFrame com colunas ['timestamp', 'open', 'high', 'low', 'close'].
    """
    try:
        url = f"{COINGECKO_OHLC_URL}{days}"
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # data é lista de [timestamp, open, high, low, close]
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        logger.info("OHLC histórico obtido: %d registros", len(df))
        return df
    except Exception as e:
        logger.exception("Erro ao buscar OHLC: %s", e)
        return None

def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Calcula RSI(14) e médias móveis (7, 25) e retorna os últimos valores.
    """
    if df is None or df.empty:
        return {}
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    # Médias móveis
    df['sma7'] = ta.trend.SMAIndicator(close=df['close'], window=7).sma_indicator()
    df['sma25'] = ta.trend.SMAIndicator(close=df['close'], window=25).sma_indicator()
    last = df.iloc[-1]
    return {
        'rsi': last.get('rsi', None),
        'sma7': last.get('sma7', None),
        'sma25': last.get('sma25', None),
        'price': last['close'],
        'timestamp': last.name,
    }

# ----------------------------------------------
# Sentiment analysis
# ----------------------------------------------
def get_sentiment_pipeline():
    """Carrega modelo de análise de sentimento (small) lazy."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline
        try:
            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=-1,  # CPU
                tokenizer="nlptown/bert-base-multilingual-uncased-sentiment"
            )
            logger.info("Modelo de sentimento carregado.")
        except Exception as e:
            logger.error("Falha ao carregar modelo de sentimento: %s", e)
            _sentiment_pipeline = False  # indica falha
    return _sentiment_pipeline if _sentiment_pipeline is not False else None

def analyze_sentiment(texts: List[str]) -> List[float]:
    """
    Retorna scores de sentimento para uma lista de textos.
    Score entre 1 (negativo) e 5 (positivo), normalizado para -1..1.
    """
    pipeline = get_sentiment_pipeline()
    if pipeline is None:
        return [0.0] * len(texts)
    try:
        results = pipeline(texts, truncation=True, max_length=512)
        # Converte rating (1 a 5) para score de -1 a 1
        scores = []
        for r in results:
            label = r['label']
            # label é algo como "1 star" ou "5 stars"
            try:
                stars = int(label.split()[0])
                score = (stars - 3) / 2  # 1-> -1, 2-> -0.5, 3->0, 4->0.5, 5->1
            except:
                score = 0.0
            scores.append(score)
        return scores
    except Exception as e:
        logger.error("Erro na análise de sentimento: %s", e)
        return [0.0] * len(texts)

# ----------------------------------------------
# Notícias com sentimento
# ----------------------------------------------
def _parse_rss(xml_text: str, max_items: int = 3) -> List[dict]:
    items = []
    try:
        root = ET.fromstring(xml_text)
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"
        channel = root.find(f"{ns}channel") or root
        entries = channel.findall(f"{ns}item") or root.findall(f"{ns}entry")
        for entry in entries[:max_items]:
            title_el = entry.find(f"{ns}title")
            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            link_el = entry.find(f"{ns}link")
            link = link_el.text if link_el is not None else link_el.get("href", "") if link_el is not None else ""
            desc_el = entry.find(f"{ns}description") or entry.find(f"{ns}summary")
            summary = ""
            if desc_el is not None and desc_el.text:
                soup = BeautifulSoup(desc_el.text, "html.parser")
                summary = soup.get_text(separator=" ", strip=True)[:300]
            pub_date_el = entry.find(f"{ns}pubDate") or entry.find(f"{ns}published") or entry.find(f"{ns}updated")
            pub_date = pub_date_el.text.strip() if pub_date_el is not None and pub_date_el.text else ""
            if title:
                items.append({"title": title, "link": link, "summary": summary, "pub_date": pub_date})
    except ET.ParseError:
        pass
    return items

from typing import List
from datetime import datetime


def fetch_bitcoin_news(max_news: int = 3) -> List[dict]:
    """
    Retorna lista de notícias agregadas de múltiplos feeds,
    com sentimento (-1 a 1), sem duplicatas e ordenadas por data.
    """
    collected = []

    # 📊 balanceamento por fonte
    per_feed = max(1, max_news // len(RSS_FEEDS))

    for feed_url in RSS_FEEDS:
        try:
            resp = requests.get(feed_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()

            items = _parse_rss(resp.text, max_items=per_feed)
            source = feed_url.split("/")[2]

            for item in items:
                item["source"] = source

                # garantir campo de data
                item["published"] = item.get("published", datetime.min)

                collected.append(item)

            logger.info("%d notícias de %s", len(items), source)

        except Exception as e:
            logger.warning("Falha no feed %s: %s", feed_url, e)

    # 🧹 remover duplicadas (por título)
    seen = set()
    unique = []

    for n in collected:
        title_key = n["title"].strip().lower()
        if title_key not in seen:
            unique.append(n)
            seen.add(title_key)

    # ⏱️ ordenar por data (mais recente primeiro)
    unique.sort(key=lambda x: x.get("published", datetime.min), reverse=True)

    # ✂️ limitar total
    final_news = unique[:max_news]

    # 🧠 sentimento
    if final_news:
        texts = [f"{n['title']} {n.get('summary','')}" for n in final_news]

        try:
            sentiments = analyze_sentiment(texts)
            for i, n in enumerate(final_news):
                n["sentiment"] = sentiments[i]
        except Exception as e:
            logger.warning("Erro ao calcular sentimento: %s", e)
            for n in final_news:
                n["sentiment"] = 0.0

    logger.info("Total final de notícias: %d", len(final_news))

    return final_news

# ----------------------------------------------
# Gráficos
# ----------------------------------------------
import numpy as np
from datetime import datetime, timedelta, timezone

def generate_price_chart(price_history: List[dict]) -> Optional[BytesIO]:
    """
    Gráfico das últimas 24h com:
    - preço BTC
    - tendência polinomial + seno
    """
    if not price_history:
        return None

    df = pd.DataFrame(price_history)

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # 🧠 filtrar últimas 24h
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)
    df = df[df['timestamp'] >= cutoff]

    if df.empty:
        return None

    df.sort_values('timestamp', inplace=True)

    # eixo numérico (tempo em horas)
    t0 = df['timestamp'].iloc[0]
    df['t'] = (df['timestamp'] - t0).dt.total_seconds() / 3600.0

    x = df['t'].values
    y = df['price_usd'].values

    fig, ax = plt.subplots(figsize=(10, 6))

    # 📈 preço real
    ax.plot(df['timestamp'], y, label='BTC', linewidth=2)

    # 🧮 tendência polinomial (grau 2)
    if len(x) >= 5:
        coeffs = np.polyfit(x, y, deg=2)
        poly = np.poly1d(coeffs)
        y_poly = poly(x)

        # 📈 seno sobre resíduo
        residuals = y - y_poly

        try:
            # frequência básica
            freq = 2 * np.pi / max(x.max(), 1e-6)
            sine = np.sin(freq * x)

            # ajustar amplitude
            amp = np.std(residuals)
            y_sine = amp * sine

            # combinar
            y_trend = y_poly + y_sine

            ax.plot(df['timestamp'], y_trend,
                    linestyle='--',
                    linewidth=2,
                    label='Tendência (poly + sine)')

        except Exception:
            # fallback só poly
            ax.plot(df['timestamp'], y_poly,
                    linestyle='--',
                    linewidth=2,
                    label='Tendência (poly)')

    # 🕒 eixo X
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    ax.set_title('Bitcoin (USD) — Últimas 24h')
    ax.set_xlabel('Hora (UTC)')
    ax.set_ylabel('Preço')

    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf

# ----------------------------------------------
# Formatação
# ----------------------------------------------
def format_price_summary(price_data: dict) -> str:
    sign = "+" if price_data["change_24h"] >= 0 else ""
    return (
        f"Bitcoin: ${price_data['price_usd']:,.2f}\n"
        f"Variação 24h: {sign}{price_data['change_24h']:.2f}%\n"
        f"Volume 24h: ${price_data['volume_24h']:,.0f}\n"
        f"Market Cap: ${price_data['market_cap']:,.0f}"
    )

def format_news_summary(news_list: list[dict]) -> str:
    if not news_list:
        return "Nenhuma notícia disponível."
    lines = ["Últimas notícias sobre Bitcoin:"]
    for i, n in enumerate(news_list, 1):
        lines.append(f"\n{i}. {n['title']}")
        if n.get('summary'):
            lines.append(f"   {n['summary'][:150]}...")
        if n.get('source'):
            lines.append(f"   Fonte: {n['source']}")
        if 'sentiment' in n:
            sent = n['sentiment']
            emoji = "🟢" if sent > 0.2 else "🔴" if sent < -0.2 else "⚪"
            lines.append(f"   Sentimento: {emoji} {sent:.2f}")
    return "\n".join(lines)

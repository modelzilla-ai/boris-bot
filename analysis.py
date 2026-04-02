"""
analysis.py -- Motor de análise com LLM melhorado
==================================================
Adicionado:
- Prompt com chain-of-thought e few-shot example
- Inclusão do score de sentimento das notícias no prompt
- Fallback com regras usando RSI e sentimento real
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    trend: str           # "Alta", "Baixa" ou "Neutra"
    recommendation: str
    confidence: str      # "Alta", "Media", "Baixa"
    reasoning: str
    used_llm: bool

_pipeline = None
_model_name: Optional[str] = None

def load_model(model_name: str) -> bool:
    global _pipeline, _model_name
    if _pipeline is not None and _model_name == model_name:
        return True
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch
        logger.info("Carregando modelo '%s' na CPU...", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        _pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
        _model_name = model_name
        logger.info("Modelo carregado.")
        return True
    except Exception as e:
        logger.error("Falha ao carregar modelo: %s", e)
        return False

def _build_prompt(price_data: dict, news_list: list[dict], price_trend_summary: str,
                  decision_summary: str, indicators: dict) -> str:
    """Prompt com chain-of-thought e exemplo few-shot."""
    sign = "+" if price_data["change_24h"] >= 0 else ""
    # Calcular sentimento médio das notícias
    avg_sentiment = sum(n.get('sentiment', 0) for n in news_list) / len(news_list) if news_list else 0
    sent_text = f"Sentimento médio das notícias: {avg_sentiment:.2f} (de -1 a 1)"
    # Notícias com sentimento
    news_text = ""
    for i, n in enumerate(news_list, 1):
        sent_emoji = "🟢" if n.get('sentiment', 0) > 0.2 else "🔴" if n.get('sentiment', 0) < -0.2 else "⚪"
        news_text += f"{i}. {n['title']} {sent_emoji}\n"
        if n.get('summary'):
            news_text += f"   Resumo: {n['summary'][:150]}\n"

    indicators_text = ""
    if indicators:
        rsi = indicators.get('rsi')
        sma7 = indicators.get('sma7')
        sma25 = indicators.get('sma25')
        if rsi is not None:
            indicators_text += f"RSI(14): {rsi:.2f} "
            if rsi > 70:
                indicators_text += "(sobrecompra) "
            elif rsi < 30:
                indicators_text += "(sobrevenda) "
        if sma7 and sma25:
            cross = "acima" if sma7 > sma25 else "abaixo"
            indicators_text += f"MM7 está {cross} da MM25. "

    prompt = f"""<|system|>
Você é Boris, um estrategista de investimentos especializado em Bitcoin. Analise os dados abaixo passo a passo e forneça sua recomendação no formato especificado.

FORMATO DE RESPOSTA OBRIGATÓRIO:
TENDÊNCIA: [Alta/Baixa/Neutra]
CONFIANÇA: [Alta/Média/Baixa]
RECOMENDAÇÃO: [Uma frase curta e direta, com nível de preço sugerido se aplicável]
ANÁLISE: [Primeiro, descreva os dados técnicos; depois, o sentimento do mercado; por fim, justifique sua tendência]

Exemplo de análise anterior:
---
TENDÊNCIA: Alta
CONFIANÇA: Média
RECOMENDAÇÃO: Mantenha posição, com alvo em $72.000.
ANÁLISE: O preço rompeu resistência de $68.000 com volume acima da média. RSI em 65 indica momentum positivo. Notícias mostram adoção institucional. Portanto, tendência de alta.
---
<|user|>
## Dados de Mercado
- Preço atual: ${price_data['price_usd']:,.2f} USD
- Variação 24h: {sign}{price_data['change_24h']:.2f}%
- Volume 24h: ${price_data['volume_24h']:,.0f} USD
- Market Cap: ${price_data['market_cap']:,.0f} USD

## Indicadores Técnicos
{indicators_text}

## Tendência Histórica
{price_trend_summary}

## Últimas Notícias
{news_text if news_text else 'Nenhuma notícia disponível.'}
{sent_text}

## Histórico de Decisões
{decision_summary}

Analise esses dados e forneça sua recomendação estratégica.
<|assistant|>
"""
    return prompt

def _parse_llm_output(text: str) -> dict:
    trend = "Neutra"
    m = re.search(r"TENDÊNCIA:\s*(Alta|Baixa|Neutra)", text, re.IGNORECASE)
    if m:
        trend = m.group(1).capitalize()
    confidence = "Média"
    m = re.search(r"CONFIANÇA:\s*(Alta|Média|Baixa)", text, re.IGNORECASE)
    if m:
        confidence = m.group(1).capitalize()
    recommendation = "Monitore o mercado."
    m = re.search(r"RECOMENDAÇÃO:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        rec = m.group(1).strip()
        if rec:
            recommendation = rec
    reasoning = ""
    m = re.search(r"ANÁLISE:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        reasoning = m.group(1).strip()
    return {"trend": trend, "confidence": confidence, "recommendation": recommendation, "reasoning": reasoning or text.strip()}

def analyze_with_llm(price_data: dict, news_list: list[dict], price_trend_summary: str,
                     decision_summary: str, model_name: str, indicators: dict = None,
                     max_new_tokens: int = 400) -> Optional[AnalysisResult]:
    if not load_model(model_name):
        return None
    try:
        prompt = _build_prompt(price_data, news_list, price_trend_summary, decision_summary, indicators or {})
        outputs = _pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=_pipeline.tokenizer.eos_token_id,
        )
        generated = outputs[0]["generated_text"]
        response_text = generated[len(prompt):].strip()
        parsed = _parse_llm_output(response_text)
        return AnalysisResult(
            trend=parsed["trend"],
            recommendation=parsed["recommendation"],
            confidence=parsed["confidence"],
            reasoning=parsed["reasoning"],
            used_llm=True,
        )
    except Exception as e:
        logger.error("Erro na análise LLM: %s", e)
        return None

def analyze_with_rules(price_data: dict, news_list: list[dict], indicators: dict = None) -> AnalysisResult:
    """Regras melhoradas com RSI e sentimento real."""
    change = price_data.get("change_24h", 0.0)
    # Sentimento médio das notícias
    sentiments = [n.get('sentiment', 0) for n in news_list] if news_list else [0]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

    rsi = indicators.get('rsi') if indicators else None

    # Decisão baseada em regras mais sofisticadas
    if change > 5.0:
        trend, confidence = "Alta", "Alta"
        rec = "Forte alta em andamento. Considere manter ou aumentar exposição gradualmente."
    elif change > 2.0:
        if avg_sentiment > 0.2 and (rsi is None or rsi < 70):
            trend, confidence = "Alta", "Média"
            rec = "Alta moderada com sentimento positivo. Mantenha posição."
        else:
            trend, confidence = "Neutra", "Média"
            rec = "Alta moderada, mas sentimento neutro ou RSI elevado. Cautela."
    elif change < -5.0:
        trend, confidence = "Baixa", "Alta"
        rec = "Forte queda. Considere reduzir exposição ou aguardar estabilização."
    elif change < -2.0:
        if avg_sentiment < -0.2 and (rsi is None or rsi > 30):
            trend, confidence = "Baixa", "Média"
            rec = "Correção com sentimento negativo. Acompanhe suportes."
        else:
            trend, confidence = "Neutra", "Média"
            rec = "Queda moderada, mas sem pânico. Mantenha posição."
    else:
        if avg_sentiment > 0.2 and (rsi is None or rsi < 70):
            trend, confidence = "Alta", "Baixa"
            rec = "Mercado lateral com notícias favoráveis. Leve tendência de alta."
        elif avg_sentiment < -0.2 and (rsi is None or rsi > 30):
            trend, confidence = "Baixa", "Baixa"
            rec = "Mercado lateral com notícias negativas. Reduza exposição."
        else:
            trend, confidence = "Neutra", "Alta"
            rec = "Mercado sem direção definida. Aguarde catalisadores."

    reasoning = f"Regras: variação {change:+.2f}%, sentimento médio {avg_sentiment:+.2f}, RSI={rsi:.2f if rsi else 'N/A'}."
    return AnalysisResult(trend, rec, confidence, reasoning, used_llm=False)

def run_analysis(price_data: dict, news_list: list[dict], price_trend_summary: str,
                 decision_summary: str, model_name: str, indicators: dict = None,
                 max_new_tokens: int = 400) -> AnalysisResult:
    result = analyze_with_llm(price_data, news_list, price_trend_summary, decision_summary,
                              model_name, indicators, max_new_tokens)
    if result:
        logger.info("LLM: tendência=%s confiança=%s", result.trend, result.confidence)
        return result
    logger.warning("LLM indisponível, usando regras.")
    return analyze_with_rules(price_data, news_list, indicators)
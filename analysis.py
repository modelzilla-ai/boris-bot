"""
analysis.py -- Motor de análise do Boris v3
============================================
Lógica de decisão completamente reescrita:
- Regras baseadas em score ponderado (change_24h + RSI + sentimento + tendência histórica)
- Tendência nunca contradiz queda com sentimento positivo isolado
- LLM como camada de refinamento, não substituição
- Confiança calculada por consenso dos sinais
"""

import logging
import re
import time
import os
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
    score: float = 0.0   # score interno [-1, 1] para debug


# ---------------------------------------------------------------------------
# Carregamento do modelo LLM
# ---------------------------------------------------------------------------
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
        logger.info("Modelo carregado com sucesso.")
        return True
    except Exception as e:
        logger.error("Falha ao carregar modelo: %s", e)
        return False


# ---------------------------------------------------------------------------
# Motor de score técnico
# ---------------------------------------------------------------------------

def _compute_technical_score(change_24h: float, rsi: Optional[float],
                               avg_sentiment: float, price_trend_pct: float) -> tuple[float, list[str]]:
    """
    Calcula um score composto no intervalo [-1, 1].

    Pesos:
      - Variação 24h:        40%
      - RSI:                 25%
      - Sentimento notícias: 20%
      - Tendência histórica: 15%

    Retorna (score, lista de sinais para raciocínio).
    """
    signals = []
    score = 0.0

    # --- Variação 24h (peso 0.40) ---
    # Mapeamento: >=+5% → +1, <=-5% → -1, linear no meio
    change_score = max(-1.0, min(1.0, change_24h / 5.0))
    score += change_score * 0.40

    direction = "alta" if change_24h >= 0 else "queda"
    signals.append(f"Variação 24h: {change_24h:+.2f}% ({direction}) → sinal {change_score:+.2f}")

    # --- RSI (peso 0.25) ---
    if rsi is not None:
        if rsi >= 70:
            rsi_score = -0.6    # sobrecompra → tendência de reversão
            rsi_label = f"sobrecompra ({rsi:.1f}) → sinal negativo"
        elif rsi <= 30:
            rsi_score = +0.6    # sobrevenda → tendência de reversão pra cima
            rsi_label = f"sobrevenda ({rsi:.1f}) → sinal positivo"
        elif rsi >= 55:
            rsi_score = +0.3    # momentum positivo
            rsi_label = f"momentum positivo ({rsi:.1f})"
        elif rsi <= 45:
            rsi_score = -0.3    # momentum negativo
            rsi_label = f"momentum negativo ({rsi:.1f})"
        else:
            rsi_score = 0.0
            rsi_label = f"neutro ({rsi:.1f})"
        score += rsi_score * 0.25
        signals.append(f"RSI(14): {rsi_label} → sinal {rsi_score:+.2f}")
    else:
        signals.append("RSI: indisponível (ignorado no score)")

    # --- Sentimento das notícias (peso 0.20) ---
    # avg_sentiment já está em [-1, 1]
    score += avg_sentiment * 0.20
    sent_label = "positivo" if avg_sentiment > 0.2 else "negativo" if avg_sentiment < -0.2 else "neutro"
    signals.append(f"Sentimento médio: {avg_sentiment:+.2f} ({sent_label}) → sinal {avg_sentiment * 0.20:+.2f}")

    # --- Tendência histórica (peso 0.15) ---
    trend_score = max(-1.0, min(1.0, price_trend_pct / 5.0))
    score += trend_score * 0.15
    trend_label = "alta" if price_trend_pct >= 0 else "queda"
    signals.append(f"Tendência histórica: {price_trend_pct:+.2f}% ({trend_label}) → sinal {trend_score * 0.15:+.2f}")

    return score, signals


def _score_to_result(score: float, signals: list[str],
                     change_24h: float, rsi: Optional[float]) -> tuple[str, str, str]:
    """
    Converte score em (tendência, recomendação, confiança).
    A confiança mede o consenso entre os sinais principais.
    """
    # --- Tendência ---
    if score >= 0.25:
        trend = "Alta"
    elif score <= -0.25:
        trend = "Baixa"
    else:
        trend = "Neutra"

    # --- Confiança (baseada na magnitude do score) ---
    abs_score = abs(score)
    if abs_score >= 0.55:
        confidence = "Alta"
    elif abs_score >= 0.30:
        confidence = "Média"
    else:
        confidence = "Baixa"

    # --- Recomendação contextualizada ---
    if trend == "Alta":
        if confidence == "Alta":
            rec = "Sinal técnico forte de alta. Considere aumentar exposição com stop definido."
        elif confidence == "Média":
            rec = "Tendência de alta com sinais moderados. Mantenha posição, monitore resistências."
        else:
            rec = "Leve tendência positiva. Aguarde confirmação antes de agir."
    elif trend == "Baixa":
        if confidence == "Alta":
            rec = "Pressão de venda intensa. Considere reduzir exposição ou proteger posição com stop."
        elif confidence == "Média":
            rec = "Tendência de queda moderada. Acompanhe suportes e evite novas entradas."
        else:
            rec = "Leve tendência negativa. Mantenha cautela e não amplie posição."
    else:
        if rsi is not None and rsi > 65:
            rec = "Mercado lateral com RSI elevado. Risco de correção — evite compras no topo."
        elif rsi is not None and rsi < 35:
            rec = "Mercado lateral com RSI baixo. Possível oportunidade de compra gradual."
        else:
            rec = "Sem direção definida. Aguarde catalisador antes de operar."

    return trend, rec, confidence


# ---------------------------------------------------------------------------
# Análise por regras (fallback ou modo padrão)
# ---------------------------------------------------------------------------

def analyze_with_rules(price_data: dict, news_list: list[dict],
                        indicators: dict = None,
                        price_trend_pct: float = 0.0) -> AnalysisResult:
    """
    Análise totalmente baseada em score técnico ponderado.
    Nunca retorna 'Alta' apenas por sentimento quando o preço está caindo.
    """
    change = price_data.get("change_24h", 0.0)
    rsi = (indicators or {}).get("rsi")

    # sentimento médio real das notícias
    sentiments = [n.get("sentiment", 0.0) for n in (news_list or [])]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

    score, signals = _compute_technical_score(change, rsi, avg_sentiment, price_trend_pct)
    trend, rec, confidence = _score_to_result(score, signals, change, rsi)

    rsi_text = f"{rsi:.1f}" if rsi is not None else "N/D"
    reasoning = (
        f"Score composto: {score:+.3f} | "
        f"Var24h: {change:+.2f}% | RSI: {rsi_text} | "
        f"Sentimento: {avg_sentiment:+.2f}\n"
        + " | ".join(signals)
    )

    logger.info("Regras → tendência=%s confiança=%s score=%.3f", trend, confidence, score)
    return AnalysisResult(trend, rec, confidence, reasoning, used_llm=False, score=score)


# ---------------------------------------------------------------------------
# Prompt para o LLM
# ---------------------------------------------------------------------------

def _build_prompt(price_data: dict, news_list: list[dict],
                   price_trend_summary: str, decision_summary: str,
                   indicators: dict, rules_result: AnalysisResult) -> str:
    """
    Prompt enriquecido com o pré-resultado das regras técnicas.
    O LLM serve para refinar/justificar, não inventar tendência.
    """
    sign = "+" if price_data["change_24h"] >= 0 else ""
    avg_sentiment = sum(n.get("sentiment", 0) for n in news_list) / len(news_list) if news_list else 0

    news_text = ""
    for i, n in enumerate(news_list, 1):
        emoji = "🟢" if n.get("sentiment", 0) > 0.2 else "🔴" if n.get("sentiment", 0) < -0.2 else "⚪"
        news_text += f"{i}. {n['title']} {emoji}\n"
        if n.get("summary"):
            news_text += f"   {n['summary'][:150]}\n"

    ind_text = ""
    if indicators:
        rsi = indicators.get("rsi")
        sma7 = indicators.get("sma7")
        sma25 = indicators.get("sma25")
        if rsi is not None:
            state = "sobrecompra" if rsi > 70 else "sobrevenda" if rsi < 30 else "neutro"
            ind_text += f"RSI(14): {rsi:.2f} ({state})\n"
        if sma7 and sma25:
            cross = "ACIMA" if sma7 > sma25 else "ABAIXO"
            ind_text += f"MM7 está {cross} da MM25 (MM7={sma7:,.0f} | MM25={sma25:,.0f})\n"

    prompt = f"""<|system|>
Você é Boris, estrategista quantitativo especializado em Bitcoin. Analise os dados abaixo com rigor técnico.

REGRA FUNDAMENTAL: A análise técnica e a variação de preço têm prioridade absoluta sobre sentimento de notícias.
Se o preço está caindo (-), a tendência NÃO pode ser "Alta" a menos que haja sinal técnico muito forte contrário (ex: RSI em sobrevenda extrema + volume crescente).

FORMATO OBRIGATÓRIO DE RESPOSTA:
TENDÊNCIA: [Alta/Baixa/Neutra]
CONFIANÇA: [Alta/Média/Baixa]
RECOMENDAÇÃO: [Uma frase direta e objetiva]
ANÁLISE: [Justificativa técnica em 2-3 frases: primeiro preço/indicadores, depois sentimento]

Exemplo correto:
---
TENDÊNCIA: Baixa
CONFIANÇA: Média
RECOMENDAÇÃO: Queda moderada com RSI ainda neutro. Evite novas compras, monitore suporte em $64.000.
ANÁLISE: Preço recuou -1.8% nas últimas 24h com momentum negativo (RSI 44). Notícias levemente positivas não compensam a pressão vendedora. Tendência de baixa com confiança média.
---
<|user|>
## Pré-análise Técnica (regras quantitativas)
Tendência calculada: {rules_result.trend}
Score composto: {rules_result.score:+.3f} (intervalo -1 a +1)
Confiança: {rules_result.confidence}
Detalhes: {rules_result.reasoning}

## Dados de Mercado
- Preço atual: ${price_data['price_usd']:,.2f} USD
- Variação 24h: {sign}{price_data['change_24h']:.2f}%
- Volume 24h: ${price_data['volume_24h']:,.0f} USD
- Market Cap: ${price_data['market_cap']:,.0f} USD

## Indicadores Técnicos
{ind_text if ind_text else 'Indisponível.'}

## Tendência Histórica
{price_trend_summary}

## Notícias Recentes
{news_text if news_text else 'Nenhuma notícia disponível.'}
Sentimento médio das notícias: {avg_sentiment:+.2f} (de -1 a +1)

## Histórico de Decisões
{decision_summary}

Refine a pré-análise com seu julgamento e responda no formato acima.
<|assistant|>
"""
    return prompt


def _parse_llm_output(text: str) -> dict:
    trend = None
    m = re.search(r"TENDÊNCIA:\s*(Alta|Baixa|Neutra)", text, re.IGNORECASE)
    if m:
        trend = m.group(1).capitalize()

    confidence = "Média"
    m = re.search(r"CONFIANÇA:\s*(Alta|Média|Baixa|Media)", text, re.IGNORECASE)
    if m:
        confidence = m.group(1).capitalize().replace("Media", "Média")

    recommendation = None
    m = re.search(r"RECOMENDAÇÃO:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        rec = m.group(1).strip()
        if rec:
            recommendation = rec

    reasoning = ""
    m = re.search(r"ANÁLISE:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        reasoning = m.group(1).strip()

    return {
        "trend": trend,
        "confidence": confidence,
        "recommendation": recommendation,
        "reasoning": reasoning or text.strip(),
    }


def analyze_with_llm(price_data: dict, news_list: list[dict],
                      price_trend_summary: str, decision_summary: str,
                      model_name: str, indicators: dict,
                      rules_result: AnalysisResult,
                      max_new_tokens: int = 400) -> Optional[AnalysisResult]:
    if not load_model(model_name):
        return None
    try:
        prompt = _build_prompt(
            price_data, news_list, price_trend_summary,
            decision_summary, indicators, rules_result
        )
        outputs = _pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.25,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=_pipeline.tokenizer.eos_token_id,
        )
        generated = outputs[0]["generated_text"]
        response_text = generated[len(prompt):].strip()
        parsed = _parse_llm_output(response_text)

        # Validação de sanidade: LLM não pode inverter tendência forte das regras
        llm_trend = parsed.get("trend")
        rules_trend = rules_result.trend
        abs_score = abs(rules_result.score)

        if llm_trend and abs_score >= 0.45 and llm_trend != rules_trend:
            logger.warning(
                "LLM sugeriu '%s' mas score técnico forte indica '%s' (score=%.3f). "
                "Usando tendência técnica.", llm_trend, rules_trend, rules_result.score
            )
            llm_trend = rules_trend  # score técnico prevalece

        return AnalysisResult(
            trend=llm_trend or rules_result.trend,
            recommendation=parsed["recommendation"] or rules_result.recommendation,
            confidence=parsed["confidence"],
            reasoning=parsed["reasoning"],
            used_llm=True,
            score=rules_result.score,
        )
    except Exception as e:
        logger.error("Erro na análise LLM: %s", e)
        return None


# ---------------------------------------------------------------------------
# Controle de uso do LLM (cooldown + gatilhos)
# ---------------------------------------------------------------------------
_last_llm_time = 0.0


def _should_use_llm(price_data: dict, indicators: dict) -> bool:
    """Aciona LLM em situações de mercado relevantes."""
    change = abs(price_data.get("change_24h", 0.0))
    if change >= 2.0:
        return True
    rsi = (indicators or {}).get("rsi")
    if rsi is not None and (rsi >= 68 or rsi <= 32):
        return True
    return False


def _cooldown_ok(cooldown_seconds: int) -> bool:
    global _last_llm_time
    now = time.time()
    if now - _last_llm_time >= cooldown_seconds:
        _last_llm_time = now
        return True
    return False


# ---------------------------------------------------------------------------
# Ponto de entrada principal
# ---------------------------------------------------------------------------

def run_analysis(price_data: dict, news_list: list[dict],
                  price_trend_summary: str, decision_summary: str,
                  model_name: str, indicators: dict = None,
                  max_new_tokens: int = 400,
                  price_trend_pct: float = 0.0) -> AnalysisResult:
    """
    Executa análise completa:
    1. Score técnico ponderado (sempre)
    2. LLM para refinamento (quando mercado justifica)
    3. Validação de sanidade entre LLM e score técnico
    """
    indicators = indicators or {}

    # 1. Análise técnica base (sempre executada)
    rules_result = analyze_with_rules(price_data, news_list, indicators, price_trend_pct)

    # 2. LLM desativado
    if model_name in (None, "", "none", "None"):
        logger.info("LLM desativado. Usando análise técnica.")
        return rules_result

    # 3. Decidir uso do LLM — mercado relevante + cooldown (sem sorteio)
    COOLDOWN = int(os.getenv("LLM_COOLDOWN_SECONDS", "9000"))  # padrão: 2h30

    use_llm = (
        _should_use_llm(price_data, indicators)
        and _cooldown_ok(COOLDOWN)
    )

    if use_llm:
        logger.info("Acionando LLM para refinamento (score_base=%.3f)", rules_result.score)
        llm_result = analyze_with_llm(
            price_data, news_list, price_trend_summary,
            decision_summary, model_name, indicators,
            rules_result, max_new_tokens
        )
        if llm_result:
            logger.info("LLM: tendência=%s confiança=%s", llm_result.trend, llm_result.confidence)
            return llm_result
        logger.warning("LLM falhou, usando análise técnica.")

    return rules_result

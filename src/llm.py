"""Seleção do provedor de LLM do Assistente Virtual do Residencial Aurora.

Precedência (definida em código, não no prompt):

1. ``GOOGLE_API_KEY`` ou ``GEMINI_API_KEY`` preenchida → Gemini nativo do ADK.
2. ``DEEPSEEK_API_KEY`` preenchida → DeepSeek via ``LiteLlm`` do ADK.
3. Nenhuma das duas → ``RuntimeError`` na subida da API.

Quando as duas chaves estão preenchidas, o Google tem preferência. O modelo de
cada provedor pode ser trocado sem alterar o código por ``GEMINI_MODEL`` e
``DEEPSEEK_MODEL``.
"""

from __future__ import annotations

import logging
import os

from google.adk.models import BaseLlm

logger = logging.getLogger(__name__)

MODELO_GOOGLE_PADRAO = "gemini-2.5-flash"
MODELO_DEEPSEEK_PADRAO = "deepseek-flash"
PREFIXO_DEEPSEEK = "deepseek/"

# O SDK do Google aceita as duas variáveis, com prioridade para GOOGLE_API_KEY.
_VARIAVEIS_GOOGLE = ("GOOGLE_API_KEY", "GEMINI_API_KEY")


def _variavel(nome: str) -> str:
    return os.getenv(nome, "").strip()


def _chave_google() -> str:
    for nome in _VARIAVEIS_GOOGLE:
        valor = _variavel(nome)
        if valor:
            return valor
    return ""


def _criar_modelo_deepseek() -> BaseLlm:
    # Import tardio: o LiteLLM só é carregado quando o DeepSeek é o provedor.
    # O mapa local de custos evita uma busca na internet a cada subida da API.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from google.adk.models.lite_llm import LiteLlm

    nome = _variavel("DEEPSEEK_MODEL") or MODELO_DEEPSEEK_PADRAO
    if "/" not in nome:
        nome = f"{PREFIXO_DEEPSEEK}{nome}"
    return LiteLlm(model=nome, api_key=_variavel("DEEPSEEK_API_KEY"))


def selecionar_modelo() -> str | BaseLlm:
    """Retorna o modelo a ser usado por todos os agentes.

    O valor devolvido é um nome de modelo Gemini (``str``, resolvido pelo ADK)
    ou uma instância de ``BaseLlm`` para o DeepSeek.

    Raises:
        RuntimeError: quando nem a chave do Google nem a do DeepSeek estão
            configuradas.
    """
    if _chave_google():
        modelo = _variavel("GEMINI_MODEL") or MODELO_GOOGLE_PADRAO
        logger.info("Provedor de LLM: Google Gemini (%s).", modelo)
        return modelo

    if _variavel("DEEPSEEK_API_KEY"):
        modelo = _criar_modelo_deepseek()
        logger.info("Provedor de LLM: DeepSeek (%s).", modelo.model)
        return modelo

    raise RuntimeError(
        "Nenhuma chave de LLM configurada. Defina GOOGLE_API_KEY (Gemini) ou "
        "DEEPSEEK_API_KEY (DeepSeek) no arquivo .env. Quando as duas estiverem "
        "preenchidas, o Gemini é usado."
    )

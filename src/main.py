"""API HTTP do Assistente Virtual do Residencial Aurora.

Contrato implementado:

* ``POST /sessoes`` → 201 ``{"session_id": "..."}``
* ``POST /sessoes/{session_id}/mensagens`` → 200
  ``{"resposta": "...", "confirmacoes_pendentes": [...]}``
* ``POST /sessoes/{session_id}/confirmacoes`` → 200 no mesmo formato;
  409 quando o id não está pendente naquela sessão.
* ``GET /sessoes/{session_id}/eventos`` → 200 com todos os eventos.
* ``GET /apartamentos/{numero}/reservas`` → 200.
* ``GET /apartamentos/{numero}/visitantes`` → 200.

Sobre a retomada de confirmação no ADK 2.2.0:

1. A tool registra a pendência usando ``tool_context.function_call_id``, que é
   o id da chamada original (por exemplo, ``reservar_area``).
2. O ADK gera um evento separado ``adk_request_confirmation`` cujo
   ``args.originalFunctionCall.id`` aponta para essa chamada original.
3. Para responder, o cliente envia o id **do evento de confirmação** em um
   ``FunctionResponse`` chamado ``adk_request_confirmation``. O
   ``Runner.run_async`` casa esse id com o evento, resolve a invocação e o
   ``_RequestConfirmationLlmRequestProcessor`` reexecuta a tool original com
   ``tool_confirmation.confirmed`` preenchido.

Por isso a API expõe o id persistido no SQLite (id original) e faz a tradução
para o id do evento ``adk_request_confirmation`` antes de chamar o Runner.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from google.adk.apps import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, Session
from google.genai import types
from pydantic import BaseModel, Field

from . import database
from .agents import criar_agente_raiz

load_dotenv(database.BASE_DIR / ".env")

APP_NAME = "residencial_aurora"
USER_ID = "morador"
SESSIONS_DB_URL = os.getenv(
    "SESSIONS_DB_URL", "sqlite+aiosqlite:///storage/sessions.db"
)

_session_service: DatabaseSessionService | None = None
_runner: Runner | None = None


# ---------------------------------------------------------------------------
# Modelos de entrada
# ---------------------------------------------------------------------------


class EntradaSessao(BaseModel):
    apartamento: str = Field(min_length=1)


class EntradaMensagem(BaseModel):
    texto: str = Field(min_length=1)


class EntradaConfirmacao(BaseModel):
    id: str = Field(min_length=1)
    confirmado: bool


# ---------------------------------------------------------------------------
# Ciclo de vida
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _session_service, _runner

    await database.inicializar_banco()

    _session_service = DatabaseSessionService(db_url=SESSIONS_DB_URL)
    agente = criar_agente_raiz()
    app_adk = App(
        name=APP_NAME,
        root_agent=agente,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    _runner = Runner(app=app_adk, session_service=_session_service)

    try:
        yield
    finally:
        if _runner is not None:
            await _runner.close()
        if _session_service is not None:
            await _session_service.close()
        _runner = None
        _session_service = None


app = FastAPI(
    title="Assistente Virtual do Residencial Aurora",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _servico() -> DatabaseSessionService:
    if _session_service is None:
        raise RuntimeError("O serviço de sessões não foi inicializado.")
    return _session_service


def _runner_atual() -> Runner:
    if _runner is None:
        raise RuntimeError("O Runner não foi inicializado.")
    return _runner


async def _obter_sessao(session_id: str) -> Session:
    sessao = await _servico().get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )
    if sessao is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sessão não encontrada.",
        )
    return sessao


async def _executar(session_id: str, mensagem: types.Content) -> list[Any]:
    eventos: list[Any] = []
    async for evento in _runner_atual().run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=mensagem,
    ):
        eventos.append(evento)
    return eventos


def _extrair_resposta(eventos: list[Any]) -> str:
    """Último texto não-vazio produzido na execução, sem partes de raciocínio."""
    resposta = ""
    for evento in eventos:
        partes = evento.content.parts if evento.content and evento.content.parts else []
        textos = [
            parte.text
            for parte in partes
            if parte.text and not getattr(parte, "thought", False)
        ]
        texto_evento = "".join(textos).strip()
        if texto_evento:
            resposta = texto_evento
        elif isinstance(getattr(evento, "output", None), str) and evento.output.strip():
            resposta = evento.output.strip()
    return resposta


def _id_confirmacao_adk(sessao: Session, confirmacao_id: str) -> str | None:
    """Traduz o id persistido no SQLite para o id do evento de confirmação.

    O banco guarda o id da chamada original da tool. O evento
    ``adk_request_confirmation`` guarda esse id em
    ``args.originalFunctionCall.id`` e tem um id próprio, que é o valor que o
    ``FunctionResponse`` precisa usar.
    """
    for evento in reversed(sessao.events):
        for chamada in evento.get_function_calls():
            if chamada.name != "adk_request_confirmation":
                continue
            argumentos = chamada.args or {}
            original = argumentos.get("originalFunctionCall")
            original_id = (
                original.get("id")
                if isinstance(original, dict)
                else getattr(original, "id", None)
            )
            if original_id == confirmacao_id and chamada.id:
                return chamada.id
    return None


async def _confirmacoes_pendentes(sessao: Session) -> list[dict[str, Any]]:
    registros = await database.listar_confirmacoes_pendentes(sessao.id)
    pendentes: list[dict[str, Any]] = []
    for registro in registros:
        if _id_confirmacao_adk(sessao, registro["id"]) is None:
            continue
        pendentes.append(
            {
                "id": registro["id"],
                "acao": registro["acao"],
                "detalhes": registro["detalhes"],
            }
        )
    return pendentes


def _resposta_confirmacao(id_adk: str, confirmado: bool) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=id_adk,
                    name="adk_request_confirmation",
                    response={"confirmed": confirmado},
                )
            )
        ],
    )


async def _resposta_conversa(session_id: str, eventos: list[Any]) -> dict[str, Any]:
    sessao = await _obter_sessao(session_id)
    return {
        "resposta": _extrair_resposta(eventos),
        "confirmacoes_pendentes": await _confirmacoes_pendentes(sessao),
    }


# ---------------------------------------------------------------------------
# Sessões, mensagens e confirmações
# ---------------------------------------------------------------------------


@app.post("/sessoes", status_code=status.HTTP_201_CREATED)
async def criar_sessao(entrada: EntradaSessao) -> dict[str, str]:
    apartamento = entrada.apartamento.strip()
    if await database.obter_apartamento(apartamento) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Apartamento não encontrado.",
        )

    sessao = await _servico().create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        state={"apartamento": apartamento},
    )
    return {"session_id": sessao.id}


@app.post("/sessoes/{session_id}/mensagens")
async def enviar_mensagem(
    session_id: str, entrada: EntradaMensagem
) -> dict[str, Any]:
    await _obter_sessao(session_id)

    mensagem = types.Content(
        role="user",
        parts=[types.Part.from_text(text=entrada.texto)],
    )
    eventos = await _executar(session_id, mensagem)
    return await _resposta_conversa(session_id, eventos)


@app.post("/sessoes/{session_id}/confirmacoes")
async def responder_confirmacao(
    session_id: str, entrada: EntradaConfirmacao
) -> dict[str, Any]:
    sessao = await _obter_sessao(session_id)

    # Primeiro localiza o evento do ADK; só depois consome a pendência no
    # SQLite. Assim uma confirmação órfã não é marcada como respondida.
    id_adk = _id_confirmacao_adk(sessao, entrada.id)
    if id_adk is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Não existe confirmação pendente com esse id nesta sessão.",
        )

    registro = await database.decidir_confirmacao(
        session_id=session_id,
        confirmacao_id=entrada.id,
        confirmado=entrada.confirmado,
    )
    if registro is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Não existe confirmação pendente com esse id nesta sessão.",
        )

    eventos = await _executar(
        session_id,
        _resposta_confirmacao(id_adk, entrada.confirmado),
    )
    return await _resposta_conversa(session_id, eventos)


# ---------------------------------------------------------------------------
# Eventos e rotas de verificação
# ---------------------------------------------------------------------------


@app.get("/sessoes/{session_id}/eventos")
async def listar_eventos(session_id: str) -> list[dict[str, Any]]:
    sessao = await _obter_sessao(session_id)
    return [
        evento.model_dump(mode="json", by_alias=True, exclude_none=True)
        for evento in sessao.events
    ]


@app.get("/apartamentos/{numero}/reservas")
async def listar_reservas(numero: str) -> list[dict[str, Any]]:
    if await database.obter_apartamento(numero) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Apartamento não encontrado.",
        )
    return await database.listar_reservas(numero)


@app.get("/apartamentos/{numero}/visitantes")
async def listar_visitantes(numero: str) -> list[dict[str, Any]]:
    if await database.obter_apartamento(numero) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Apartamento não encontrado.",
        )
    return await database.listar_visitantes(numero)

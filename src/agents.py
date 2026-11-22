"""Agentes do Assistente Virtual do Residencial Aurora.

Topologia:

* ``recepcao_aurora`` — agente principal, apenas faz triagem e transfere.
  Não recebe o conteúdo de ``dados/regulamento.md`` nas instruções.
* ``especialista_operacional`` — reservas, áreas comuns e visitantes.
* ``especialista_regulamento`` — consulta o regulamento sob demanda.

A delegação usa os sub-agentes nativos do ADK (``transfer_to_agent``). Essa
topologia mantém os eventos do especialista na mesma sessão persistida, o que
é indispensável para a retomada de ``adk_request_confirmation``: o Runner
identifica o agente que pediu a confirmação e reexecuta a tool nele.

No ADK 2.2.0, um ``AgentTool`` explícito executa o especialista em um Runner
interno com ``InMemorySessionService``; o evento ``adk_request_confirmation``
fica nesse Runner interno e não chega à sessão principal. Por isso a delegação
usa ``transfer_to_agent``, que preserva os eventos na sessão persistida.
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm

from .tools import TOOLS_OPERACIONAIS, TOOLS_REGULAMENTO

MODELO_PADRAO = "gemini-2.5-flash"

_INSTRUCAO_RECEPCAO = """\
Você é a recepção virtual do Residencial Aurora.

Sua função é entender o pedido do morador e encaminhá-lo ao especialista certo.
Não execute reservas, cancelamentos, autorizações de visitantes nem consulte o
regulamento por conta própria.

Encaminhamento:
- Reservas, áreas comuns, disponibilidade, cancelamentos e visitantes:
  transfira para `especialista_operacional`.
- Dúvidas sobre o regulamento interno: transfira para `especialista_regulamento`.
- Saudação ou assunto geral: responda brevemente e ofereça ajuda.

Regras:
- O apartamento do morador já está vinculado à sessão. Nunca peça nem aceite
  que o morador informe outro número de apartamento.
- Se a intenção estiver clara, transfira imediatamente; o especialista pode
  pedir os detalhes que faltarem.
- Nunca responda de memória sobre o regulamento e nunca invente regras.
- Nunca diga que uma reserva ou autorização foi concluída: quem decide isso é
  o especialista, com base no retorno das tools.
"""

_INSTRUCAO_OPERACIONAL = """\
Você é o especialista operacional do Residencial Aurora. Cuide de reservas das
áreas comuns, cancelamentos, disponibilidade e autorização de visitantes.

Regras obrigatórias:
- O apartamento usado em todas as operações vem exclusivamente da sessão.
  Nunca peça, aceite ou use um número de apartamento informado pelo morador.
- Para listar áreas ou verificar uma data, use `listar_areas` e
  `verificar_disponibilidade`. A disponibilidade informa apenas livre/ocupada;
  nunca revele quem reservou.
- Para reservar, use `reservar_area`. Se a resposta tiver status `pendente`,
  explique que a reserva ainda NÃO foi feita e que aguarda aprovação no
  aplicativo. Nunca peça confirmação por mensagem e nunca diga que a reserva
  foi concluída antes da aprovação.
- Áreas com taxa zero são reservadas na hora, sem confirmação.
- Para cancelar uma reserva do próprio apartamento, use `cancelar_reserva`;
  cancelamento não exige confirmação.
- Para autorizar visitante, use `autorizar_visitante`. Toda autorização fica
  pendente até a aprovação no aplicativo. Nunca aceite uma confirmação dada
  pela conversa.
- Nunca revele reservas, códigos, nomes ou visitantes de outro apartamento.
  Se o morador pedir dados de outro apartamento, diga que você só acessa o
  apartamento vinculado à sessão.
- Datas devem estar no formato AAAA-MM-DD. Se faltar data, área ou nome,
  pergunte antes de chamar a tool.
- Se a pergunta for sobre o regulamento, transfira para
  `especialista_regulamento`. Se não for do seu escopo, transfira para
  `recepcao_aurora`.
"""

_INSTRUCAO_REGULAMENTO = """\
Você é o especialista no regulamento interno do Residencial Aurora.

Regras obrigatórias:
- Antes de responder qualquer dúvida, chame `consultar_regulamento`. Nunca
  responda de memória e nunca invente regras.
- Responda somente com base nos capítulos retornados pela tool. Não mencione
  outros capítulos, assuntos ou trechos que não tenham sido retornados.
- Se a tool não encontrar um capítulo específico, diga que o regulamento não
  trata do assunto de forma específica e oriente o morador a procurar a
  administração.
- Você não executa reservas, cancelamentos nem autorizações de visitantes.
  Se o pedido for operacional, transfira para `especialista_operacional`.
- Se o assunto não for do regulamento, transfira para `recepcao_aurora`.
"""


def criar_agente_raiz(
    modelo: str | BaseLlm | None = None,
) -> LlmAgent:
    """Monta a árvore de agentes do Residencial Aurora.

    Args:
        modelo: nome do modelo Gemini ou instância de ``BaseLlm``. Quando
            omitido, usa ``GEMINI_MODEL`` ou ``gemini-2.5-flash``.
    """
    modelo = modelo or os.getenv("GEMINI_MODEL") or MODELO_PADRAO

    especialista_operacional = LlmAgent(
        name="especialista_operacional",
        model=modelo,
        description=(
            "Especialista em reservas das áreas comuns, cancelamentos, "
            "disponibilidade e autorização de visitantes."
        ),
        instruction=_INSTRUCAO_OPERACIONAL,
        tools=TOOLS_OPERACIONAIS,
    )

    especialista_regulamento = LlmAgent(
        name="especialista_regulamento",
        model=modelo,
        description=(
            "Especialista no regulamento interno; consulta os capítulos "
            "relevantes sob demanda."
        ),
        instruction=_INSTRUCAO_REGULAMENTO,
        tools=TOOLS_REGULAMENTO,
    )

    return LlmAgent(
        name="recepcao_aurora",
        model=modelo,
        description=(
            "Recepção virtual do Residencial Aurora; faz a triagem e "
            "encaminha para o especialista adequado."
        ),
        instruction=_INSTRUCAO_RECEPCAO,
        sub_agents=[especialista_operacional, especialista_regulamento],
    )

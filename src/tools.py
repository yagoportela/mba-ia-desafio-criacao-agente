"""Tools do Assistente Virtual do Residencial Aurora.

As tools operacionais leem o apartamento **exclusivamente** de
``tool_context.state["apartamento"]``. Nenhuma delas aceita um apartamento
escolhido pelo modelo, o que implementa a Garantia 2 mesmo diante de prompt
injection.

A Garantia 1 fica concentrada em ``reservar_area`` e ``autorizar_visitante``:

* ``autorizar_visitante`` sempre pede confirmação;
* ``reservar_area`` só pede confirmação quando a taxa da área é maior que zero;
* o pedido é registrado em ``storage/condominio.db`` com o id da chamada da
  tool, e a gravação de negócio só acontece quando
  ``tool_context.tool_confirmation.confirmed`` é ``True``.

``consultar_regulamento`` lê ``dados/regulamento.md``, divide o texto em
seções ``##`` e devolve somente os capítulos pontuados como relevantes para a
consulta — nunca o arquivo inteiro.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from datetime import date
from typing import Any

from google.adk.tools import ToolContext

from . import database

_REGULAMENTO_PATH = database.DADOS_DIR / "regulamento.md"

_FORMATO_DATA = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TITULO_SECAO = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

_MSG_OCUPADO = (
    "Não foi possível reservar: a área já está ocupada nessa data. "
    "Escolha outra data ou outra área."
)
_MSG_PENDENTE = (
    "A ação ainda não foi executada. Ela ficará pendente até o morador "
    "aprovar a confirmação no aplicativo. Não peça confirmação por mensagem "
    "e não execute a ação sem a aprovação do sistema."
)
_MSG_NEGADA = "A confirmação foi negada. Nada foi gravado."


# ---------------------------------------------------------------------------
# Helpers de sessão, área e data
# ---------------------------------------------------------------------------


def _apartamento_da_sessao(tool_context: ToolContext) -> str:
    """Lê o apartamento do state da sessão, nunca de um argumento do modelo."""
    try:
        apartamento = tool_context.state["apartamento"]
    except KeyError as exc:
        raise ValueError(
            "A sessão não está vinculada a um apartamento. "
            "Crie a sessão informando o apartamento do morador."
        ) from exc

    if not isinstance(apartamento, str) or not apartamento.strip():
        raise ValueError(
            "A sessão não está vinculada a um apartamento válido. "
            "Crie a sessão informando o apartamento do morador."
        )
    return apartamento.strip()


def _id_confirmacao(tool_context: ToolContext) -> str:
    """Id estável da chamada da tool, usado para persistir a pendência.

    O ADK gera depois um evento ``adk_request_confirmation`` que carrega este
    mesmo id em ``args.originalFunctionCall.id``. A API usa essa relação para
    ligar a resposta do cliente à retomada correta da tool.
    """
    if not tool_context.function_call_id:
        raise ValueError(
            "Não foi possível identificar a chamada da tool para registrar "
            "a confirmação."
        )
    return tool_context.function_call_id


def _normalizar_data(data: str) -> str:
    data = (data or "").strip()
    if _FORMATO_DATA.fullmatch(data) is None:
        raise ValueError("Data inválida. Use o formato AAAA-MM-DD.")
    try:
        date.fromisoformat(data)
    except ValueError as exc:
        raise ValueError("Data inválida. Use o formato AAAA-MM-DD.") from exc
    return data


def _reais(valor: float) -> str:
    return f"R$ {valor:.2f}".replace(".", ",")


def _normalizar(texto: str) -> str:
    decomposto = unicodedata.normalize("NFKD", texto)
    sem_acentos = "".join(
        caractere for caractere in decomposto if not unicodedata.combining(caractere)
    )
    return sem_acentos.casefold().strip()


def _slug(texto: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _normalizar(texto)).strip("-")


_ALIASES_AREA = {
    "salao": "salao-de-festas",
    "salao de festa": "salao-de-festas",
    "salao de festas": "salao-de-festas",
    "churrasco": "churrasqueira",
    "quadra de esportes": "quadra",
}


async def _resolver_area(area: str) -> dict[str, Any] | None:
    """Aceita o id ou o nome falado da área e devolve o registro do banco."""
    area = (area or "").strip()
    if not area:
        return None

    for candidato in (area, area.lower(), _slug(area)):
        encontrada = await database.obter_area(candidato)
        if encontrada is not None:
            return encontrada

    alias = _ALIASES_AREA.get(_normalizar(area))
    if alias is not None:
        encontrada = await database.obter_area(alias)
        if encontrada is not None:
            return encontrada

    alvo = _normalizar(area)
    for item in await database.listar_areas():
        nome = _normalizar(item["nome"])
        if alvo == nome or alvo in nome or nome in alvo:
            return item
    return None


# ---------------------------------------------------------------------------
# Tools de áreas e reservas
# ---------------------------------------------------------------------------


async def listar_areas() -> dict[str, Any]:
    """Lista as áreas comuns, seus ids e a taxa de reserva em reais.

    Use o ``id`` retornado ao chamar as demais tools. Área com taxa ``0`` não
    gera cobrança e não exige confirmação para reservar.
    """
    areas = await database.listar_areas()
    return {
        "areas": [
            {
                "id": area["id"],
                "nome": area["nome"],
                "taxa": float(area["taxa"]),
                "gera_cobranca": float(area["taxa"]) > 0,
            }
            for area in areas
        ]
    }


async def verificar_disponibilidade(
    area: str, data: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Informa apenas se uma área está livre ou ocupada em determinada data.

    A resposta contém somente o status da agenda. Nunca informa quem reservou,
    o apartamento ou o código de outra reserva.

    Args:
        area: id ou nome da área comum (ex.: "salao-de-festas").
        data: data desejada no formato AAAA-MM-DD.
    """
    _apartamento_da_sessao(tool_context)

    try:
        data = _normalizar_data(data)
    except ValueError as exc:
        return {
            "disponivel": False,
            "status": "data_invalida",
            "mensagem": str(exc),
        }

    dados_area = await _resolver_area(area)
    if dados_area is None:
        return {
            "disponivel": False,
            "status": "area_inexistente",
            "mensagem": "Área comum não encontrada.",
        }

    try:
        livre = await database.verificar_disponibilidade(dados_area["id"], data)
    except ValueError as exc:
        return {
            "disponivel": False,
            "status": "data_invalida",
            "mensagem": str(exc),
        }

    return {
        "area": dados_area["id"],
        "data": data,
        "disponivel": livre,
        "status": "livre" if livre else "ocupada",
    }


async def reservar_area(
    area: str, data: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Reserva uma área comum para o apartamento vinculado à sessão.

    O apartamento vem exclusivamente da sessão autenticada; nunca aceite um
    apartamento informado pelo morador. Áreas com taxa maior que zero só são
    gravadas depois da confirmação do sistema. Se o retorno tiver status
    ``pendente``, explique que a reserva aguarda aprovação no aplicativo.

    Args:
        area: id ou nome da área comum (ex.: "salao-de-festas").
        data: data da reserva no formato AAAA-MM-DD.
    """
    apartamento = _apartamento_da_sessao(tool_context)

    try:
        data = _normalizar_data(data)
    except ValueError as exc:
        return {"sucesso": False, "mensagem": str(exc)}

    dados_area = await _resolver_area(area)
    if dados_area is None:
        return {"sucesso": False, "mensagem": "Área comum não encontrada."}

    try:
        disponivel = await database.verificar_disponibilidade(dados_area["id"], data)
    except ValueError as exc:
        return {"sucesso": False, "mensagem": str(exc)}

    if not disponivel:
        return {"sucesso": False, "mensagem": _MSG_OCUPADO}

    taxa = float(dados_area["taxa"])
    confirmacao_id: str | None = None

    # Garantia 1: só áreas com cobrança pedem confirmação.
    if taxa > 0:
        confirmacao_id = _id_confirmacao(tool_context)
        confirmacao = tool_context.tool_confirmation

        if confirmacao is None:
            detalhes = {
                "area": dados_area["id"],
                "area_nome": dados_area["nome"],
                "data": data,
                "taxa": taxa,
            }
            await database.registrar_confirmacao(
                session_id=tool_context.session.id,
                confirmacao_id=confirmacao_id,
                tipo="reserva",
                acao="reservar_area",
                detalhes=detalhes,
            )
            tool_context.request_confirmation(
                hint=(
                    f"Confirme a reserva de {dados_area['nome']} em {data}, "
                    f"com taxa de {_reais(taxa)}."
                ),
                payload=detalhes,
            )
            tool_context.actions.skip_summarization = True
            return {
                "sucesso": False,
                "status": "pendente",
                "requer_confirmacao": True,
                "mensagem": _MSG_PENDENTE,
            }

        if not confirmacao.confirmed:
            return {
                "sucesso": False,
                "status": "negada",
                "mensagem": _MSG_NEGADA,
            }

    resultado = await database.criar_reserva(apartamento, dados_area["id"], data)

    if confirmacao_id is not None:
        await database.registrar_resultado_confirmacao(confirmacao_id, resultado)

    return resultado


async def cancelar_reserva(
    tool_context: ToolContext,
    codigo: str = "",
    area: str = "",
    data: str = "",
) -> dict[str, Any]:
    """Cancela uma reserva ativa do próprio apartamento, sem confirmação.

    Informe o código da reserva ou, alternativamente, a área e a data. A tool
    só enxerga reservas do apartamento vinculado à sessão e não pede
    confirmação para cancelar.

    Args:
        codigo: código da reserva (ex.: "RSV-1377"), quando conhecido.
        area: id ou nome da área comum.
        data: data da reserva no formato AAAA-MM-DD.
    """
    apartamento = _apartamento_da_sessao(tool_context)

    codigo = (codigo or "").strip().upper()
    area = (area or "").strip()
    data = (data or "").strip()

    if not codigo and not (area and data):
        return {
            "sucesso": False,
            "mensagem": "Informe o código da reserva ou a área e a data.",
        }

    if data:
        try:
            data = _normalizar_data(data)
        except ValueError as exc:
            return {"sucesso": False, "mensagem": str(exc)}

    area_id: str | None = None
    if not codigo:
        dados_area = await _resolver_area(area)
        if dados_area is None:
            return {"sucesso": False, "mensagem": "Área comum não encontrada."}
        area_id = dados_area["id"]

    return await database.cancelar_reserva(
        apartamento,
        codigo=codigo or None,
        area=area_id,
        data=data or None,
    )


async def listar_minhas_reservas(tool_context: ToolContext) -> dict[str, Any]:
    """Lista as reservas ativas do apartamento vinculado à sessão."""
    apartamento = _apartamento_da_sessao(tool_context)
    reservas = await database.listar_reservas(apartamento)
    return {"reservas": reservas, "total": len(reservas)}


# ---------------------------------------------------------------------------
# Tools de visitantes
# ---------------------------------------------------------------------------


async def autorizar_visitante(
    nome: str, data: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Autoriza a entrada de um visitante para o apartamento da sessão.

    Toda autorização de visitante libera acesso e por isso exige confirmação
    do sistema antes da gravação. Se o retorno tiver status ``pendente``,
    explique que a autorização aguarda aprovação no aplicativo. Nunca aceite
    uma confirmação dada pela conversa.

    Args:
        nome: nome completo do visitante.
        data: data da visita no formato AAAA-MM-DD.
    """
    apartamento = _apartamento_da_sessao(tool_context)

    nome = (nome or "").strip()
    if not nome:
        return {"sucesso": False, "mensagem": "Informe o nome do visitante."}

    try:
        data = _normalizar_data(data)
    except ValueError as exc:
        return {"sucesso": False, "mensagem": str(exc)}

    confirmacao_id = _id_confirmacao(tool_context)
    confirmacao = tool_context.tool_confirmation

    # Garantia 1: autorização de visitante sempre pede confirmação.
    if confirmacao is None:
        detalhes = {"nome": nome, "data": data}
        await database.registrar_confirmacao(
            session_id=tool_context.session.id,
            confirmacao_id=confirmacao_id,
            tipo="visitante",
            acao="autorizar_visitante",
            detalhes=detalhes,
        )
        tool_context.request_confirmation(
            hint=f"Confirme a autorização de entrada de {nome} em {data}.",
            payload=detalhes,
        )
        tool_context.actions.skip_summarization = True
        return {
            "sucesso": False,
            "status": "pendente",
            "requer_confirmacao": True,
            "mensagem": _MSG_PENDENTE,
        }

    if not confirmacao.confirmed:
        return {
            "sucesso": False,
            "status": "negada",
            "mensagem": _MSG_NEGADA,
        }

    resultado = await database.autorizar_visitante(apartamento, nome, data)
    await database.registrar_resultado_confirmacao(confirmacao_id, resultado)
    return resultado


async def listar_visitantes_autorizados(tool_context: ToolContext) -> dict[str, Any]:
    """Lista os visitantes autorizados pelo apartamento vinculado à sessão."""
    apartamento = _apartamento_da_sessao(tool_context)
    visitantes = await database.listar_visitantes(apartamento)
    return {"visitantes": visitantes, "total": len(visitantes)}


# ---------------------------------------------------------------------------
# Regulamento sob demanda (Garantia 4)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        "a", "o", "as", "os", "um", "uma", "uns", "umas",
        "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
        "por", "para", "pra", "com", "sem", "e", "ou", "que", "se",
        "ao", "aos", "eu", "meu", "minha", "meus", "minhas", "mim",
        "posso", "pode", "podemos", "poderia", "gostaria", "queria", "quero",
        "saber", "qual", "quais", "quando", "quanto", "quantos", "quantas",
        "onde", "como", "eh", "sao", "ser", "esta", "estao", "estou",
        "tem", "ter", "ha", "sobre", "mais", "menos", "ate", "depois",
        "antes", "durante", "favor", "entao", "aqui", "isso", "isto",
        "aquilo", "regulamento", "regra", "regras", "condominio",
        "residencial", "aurora", "morador", "moradores", "predio", "edificio",
        "informacao", "informacoes", "duvida", "duvidas", "area", "areas",
        "comum", "comuns", "data", "datas", "nome", "cor", "cores",
        "capitulo", "capitulos", "artigo", "artigos",
    }
)

_TERMOS_GENERICOS = frozenset(
    {
        "hora", "horario", "funciona", "funcionamento", "funcionar",
        "abre", "abrir", "fecha", "fechar", "periodo", "dia",
        "uso", "usar", "utilizar", "utilizacao", "permitido", "proibido",
        "regra", "questao",
    }
)

_SINONIMOS: dict[str, tuple[str, ...]] = {
    "barulho": ("ruido", "silencio", "sossego"),
    "ruido": ("barulho", "silencio", "sossego"),
    "silencio": ("ruido", "barulho", "sossego"),
    "som": ("ruido", "musica", "volume", "silencio"),
    "festa": ("salao", "evento", "confraternizacao"),
    "salao": ("festa", "evento", "confraternizacao"),
    "cachorro": ("cao", "animal", "pet"),
    "cao": ("cachorro", "animal", "pet"),
    "pet": ("animal", "cao", "cachorro"),
    "animal": ("cao", "cachorro", "pet"),
    "carro": ("veiculo", "garagem", "estacionamento", "estacionar"),
    "veiculo": ("carro", "garagem", "estacionamento"),
    "moto": ("motocicleta", "garagem", "veiculo"),
    "bicicleta": ("bicicletario", "garagem"),
    "visita": ("visitante", "convidado", "portaria"),
    "visitante": ("visita", "convidado", "portaria"),
    "convidado": ("visita", "visitante"),
    "entrega": ("encomenda", "portaria"),
    "encomenda": ("entrega", "portaria"),
    "mudanca": ("volumoso", "mudar"),
    "obra": ("reforma",),
    "reforma": ("obra",),
    "lixo": ("residuo", "coleta", "reciclagem", "descarte"),
    "residuo": ("lixo", "coleta", "reciclagem"),
    "reciclagem": ("lixo", "residuo", "coleta"),
    "multa": ("penalidade", "infracao"),
    "penalidade": ("multa", "infracao"),
    "infracao": ("multa", "penalidade"),
    "estacionar": ("garagem", "vaga", "veiculo"),
    "vaga": ("garagem", "estacionamento"),
    "seguranca": ("portaria",),
    "portaria": ("seguranca", "visitante", "entrega"),
    "academia": ("exercicio", "aparelho"),
    "piscina": ("deck",),
    "churrasqueira": ("churrasco",),
    "quadra": ("esporte", "poliesportiva"),
    "brinquedoteca": ("brinquedo", "crianca"),
    "playground": ("brinquedo", "crianca"),
}


def _singular(palavra: str) -> str:
    """Normalização leve de plural, suficiente para o vocabulário do texto."""
    if palavra.endswith("coes") and len(palavra) > 5:
        return palavra[:-4] + "cao"
    if palavra.endswith("ais") and len(palavra) > 4:
        return palavra[:-3] + "al"
    if palavra.endswith("aes") and len(palavra) > 4:
        return palavra[:-3] + "ao"
    if palavra.endswith("oes") and len(palavra) > 4:
        return palavra[:-3] + "ao"
    if palavra.endswith("s") and len(palavra) > 3:
        return palavra[:-1]
    return palavra


def _palavras(texto: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalizar(texto))


def _indexar(texto: str) -> Counter[str]:
    indice: Counter[str] = Counter()
    for palavra in _palavras(texto):
        indice[palavra] += 1
        singular = _singular(palavra)
        if singular != palavra:
            indice[singular] += 1
    return indice


def _termos_da_consulta(consulta: str) -> list[str]:
    termos: list[str] = []
    for palavra in _palavras(consulta):
        candidato = _singular(palavra)
        if (
            len(candidato) > 2
            and candidato not in _STOPWORDS
            and candidato not in termos
        ):
            termos.append(candidato)
    return termos


def _variantes(termo: str) -> set[str]:
    variantes = {termo, _singular(termo)}
    for chave in (termo, _singular(termo)):
        variantes.update(_SINONIMOS.get(chave, ()))
    return {
        _singular(_normalizar(variante))
        for variante in variantes
        if variante
    }


def _dividir_secoes(texto: str) -> list[tuple[str, str]]:
    partes = _TITULO_SECAO.split(texto)
    secoes: list[tuple[str, str]] = []
    for indice in range(1, len(partes) - 1, 2):
        titulo = partes[indice].strip()
        corpo = partes[indice + 1].strip()
        if titulo and corpo:
            secoes.append((titulo, corpo))
    return secoes


def _pontuar_secao(
    titulo: str, corpo: str, termos: list[str]
) -> tuple[float, bool]:
    palavras_titulo = set(_palavras(titulo))
    palavras_titulo.update(_singular(palavra) for palavra in list(palavras_titulo))

    indice_corpo = _indexar(corpo)
    pontos = 0.0
    tem_titulo = False
    topicos_no_corpo: set[str] = set()
    maior_frequencia = 0

    for termo in termos:
        variantes = _variantes(termo)
        generico = (
            termo in _TERMOS_GENERICOS or _singular(termo) in _TERMOS_GENERICOS
        )
        no_titulo = bool(variantes & palavras_titulo)
        ocorrencias = sum(indice_corpo[variante] for variante in variantes)

        if no_titulo:
            if generico:
                pontos += 1.0
            else:
                pontos += 10.0
                tem_titulo = True

        if ocorrencias:
            fator = 0.5 if generico else 2.0
            pontos += min(ocorrencias, 4) * fator
            if not generico:
                topicos_no_corpo.add(termo)
                maior_frequencia = max(maior_frequencia, ocorrencias)

    # Sem casar com o título, uma única menção isolada é evidência fraca:
    # evita devolver um capítulo inteiro por causa de uma palavra genérica que
    # aparece de passagem em um artigo de outro assunto.
    if not tem_titulo and len(topicos_no_corpo) < 2 and maior_frequencia < 2:
        return 0.0, False

    return pontos, tem_titulo


def consultar_regulamento(consulta: str) -> dict[str, Any]:
    """Consulta o regulamento interno e devolve somente os capítulos relevantes.

    A busca acontece por seção (títulos ``##``) e o retorno contém apenas os
    capítulos relacionados ao assunto perguntado. Nunca devolve o regulamento
    inteiro. Se nenhum capítulo tratar do assunto, ``encontrado`` é ``False``.

    Args:
        consulta: assunto ou pergunta do morador (ex.: "horário da piscina").
    """
    consulta = (consulta or "").strip()
    if not consulta:
        return {
            "encontrado": False,
            "consulta": consulta,
            "capitulos": [],
            "mensagem": "Informe o assunto que deseja consultar no regulamento.",
        }

    try:
        texto = _REGULAMENTO_PATH.read_text(encoding="utf-8")
    except OSError:
        return {
            "encontrado": False,
            "consulta": consulta,
            "capitulos": [],
            "mensagem": "O regulamento não está disponível no momento.",
        }

    termos = _termos_da_consulta(consulta)
    if not termos:
        return {
            "encontrado": False,
            "consulta": consulta,
            "capitulos": [],
            "mensagem": (
                "A consulta não indica um assunto específico do regulamento. "
                "Pergunte ao morador qual tema ele quer verificar."
            ),
        }

    pontuadas: list[tuple[float, int, str, str, bool]] = []
    for ordem, (titulo, corpo) in enumerate(_dividir_secoes(texto)):
        pontos, tem_titulo = _pontuar_secao(titulo, corpo, termos)
        if pontos > 0:
            pontuadas.append((pontos, ordem, titulo, corpo, tem_titulo))

    if not pontuadas:
        return {
            "encontrado": False,
            "consulta": consulta,
            "capitulos": [],
            "mensagem": (
                "Nenhum capítulo do regulamento trata desse assunto. "
                "Oriente o morador a procurar a administração."
            ),
        }

    pontuadas.sort(key=lambda item: (-item[0], item[1]))
    melhor = pontuadas[0][0]

    if melhor < 2.0:
        return {
            "encontrado": False,
            "consulta": consulta,
            "capitulos": [],
            "mensagem": (
                "Nenhum capítulo do regulamento trata desse assunto de forma "
                "específica. Oriente o morador a procurar a administração."
            ),
        }

    selecionadas = [pontuadas[0]]

    # Um segundo capítulo só entra se também tiver casado com o título e for
    # próximo do melhor resultado; isso evita despejar capítulos de outros
    # assuntos no contexto da sessão.
    if len(pontuadas) > 1:
        candidata = pontuadas[1]
        if candidata[4] and candidata[0] >= max(2.0, melhor * 0.5):
            selecionadas.append(candidata)

    return {
        "encontrado": True,
        "consulta": consulta,
        "capitulos": [
            {"titulo": titulo, "texto": corpo}
            for _, _, titulo, corpo, _ in selecionadas
        ],
    }


# ---------------------------------------------------------------------------
# Registro das tools para os agentes
# ---------------------------------------------------------------------------


TOOLS_OPERACIONAIS = [
    listar_areas,
    verificar_disponibilidade,
    reservar_area,
    cancelar_reserva,
    listar_minhas_reservas,
    autorizar_visitante,
    listar_visitantes_autorizados,
]

TOOLS_REGULAMENTO = [consultar_regulamento]

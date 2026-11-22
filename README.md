# Assistente Virtual do Residencial Aurora

Assistente multiagente para moradores do Residencial Aurora, construído com Google ADK 2.2.0 e exposto por uma API FastAPI. O modelo conduz a conversa; as regras críticas — confirmação de cobrança/acesso, isolamento por apartamento, persistência, consulta sob demanda ao regulamento e exclusividade de reservas — são aplicadas por código e pelo SQLite, não pelo que o modelo decide.

Stack: Python 3.12+, `uv`, FastAPI, Uvicorn, Google ADK 2.2.0, SQLite (`aiosqlite`) e `litellm` para o provedor DeepSeek. O provedor é escolhido em `src/llm.py`: Gemini quando `GOOGLE_API_KEY` está preenchida, DeepSeek quando apenas `DEEPSEEK_API_KEY` está preenchida, e erro na subida quando nenhuma existe. Com as duas, Gemini tem preferência.

## Arquitetura

A árvore é montada por `criar_agente_raiz()` em `src/agents.py`. Os três agentes compartilham o mesmo modelo, resolvido por `selecionar_modelo()` em `src/llm.py`.

**Agente principal — `recepcao_aurora`.** Faz apenas a triagem e a transferência. Não possui tools de negócio e não recebe o conteúdo de `dados/regulamento.md` nas instruções. Sua instrução (`_INSTRUCAO_RECEPCAO`) orienta a encaminhar reservas, visitantes e disponibilidade para o especialista operacional, e dúvidas de regulamento para o especialista de regulamento.

**Especialista operacional — `especialista_operacional`.** Concentra as tools que leem e gravam dados do condomínio: áreas, disponibilidade, reservas, cancelamentos e visitantes. É o único agente com permissão de escrita no banco de negócio.

**Especialista de regulamento — `especialista_regulamento`.** Tem apenas a tool `consultar_regulamento`, que lê `dados/regulamento.md` sob demanda e devolve somente os capítulos relevantes.

As tools são registradas em `src/tools.py` e distribuídas assim:

```python
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
```

A árvore é construída com `sub_agents`, o que faz o ADK expor a tool nativa `transfer_to_agent` ao agente principal:

```python
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
```

**Por que essa separação.** O agente principal não tem as tools de escrita, então o modelo não consegue reservar nem autorizar visitante diretamente. O regulamento fica fora do contexto da recepção e do especialista operacional: só entra na conversa o trecho retornado por `consultar_regulamento`. Isso reduz o privilégio de cada agente e mantém o histórico da sessão sem capítulos inteiros do regulamento.

**Acionamento e retomada de confirmação.** A transferência usa os sub-agentes nativos do ADK (`transfer_to_agent`). Essa escolha é deliberada: em um `AgentTool` explícito, o especialista executaria em um Runner interno com `InMemorySessionService`, e o evento `adk_request_confirmation` ficaria nesse Runner interno, fora da sessão principal. Com sub-agentes nativos, os eventos do especialista permanecem na mesma sessão persistida, e o Runner consegue identificar qual agente pediu a confirmação e reexecutar a tool nele — inclusive depois de reiniciar a API.

O docstring do módulo registra a decisão:

```text
A delegação usa os sub-agentes nativos do ADK (``transfer_to_agent``). Essa
topologia mantém os eventos do especialista na mesma sessão persistida, o que
é indispensável para a retomada de ``adk_request_confirmation``: o Runner
identifica o agente que pediu a confirmação e reexecuta a tool nele.

No ADK 2.2.0, um ``AgentTool`` explícito executa o especialista em um Runner
interno com ``InMemorySessionService``; o evento ``adk_request_confirmation``
fica nesse Runner interno e não chega à sessão principal. Por isso a delegação
usa ``transfer_to_agent``, que preserva os eventos na sessão persistida.
```

O fluxo de retomada é:

1. A tool registra a pendência com o id da chamada original (`tool_context.function_call_id`) e chama `tool_context.request_confirmation(...)`.
2. O ADK emite um evento `adk_request_confirmation` cujo `args.originalFunctionCall.id` aponta para essa chamada original.
3. A API traduz o id persistido no SQLite para o id do evento `adk_request_confirmation` com `_id_confirmacao_adk`.
4. `POST /sessoes/{session_id}/confirmacoes` envia um `FunctionResponse` chamado `adk_request_confirmation`; o `Runner` casa o id, resolve a invocação e reexecuta a tool original no agente que pediu a confirmação, agora com `tool_confirmation.confirmed` preenchido.

A resposta enviada ao Runner é montada em `src/main.py`:

```python
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
```

## Garantias

### Garantia 1 — Cobrança ou acesso só com confirmação

**Arquivo:** `src/tools.py`, função `reservar_area`. A confirmação só é pedida quando a área tem `taxa > 0`; a gravação de negócio fica depois da verificação de `tool_confirmation.confirmed`.

```python
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
```

**Arquivo:** `src/tools.py`, função `autorizar_visitante`. Toda autorização de visitante pede confirmação, independentemente de o morador dizer na conversa que já confirmou.

```python
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
```

**Arquivo:** `src/database.py`, schema da tabela `confirmacoes`. Cada pendência pertence a uma sessão e tem estado controlado; a resposta final é registrada para auditoria.

```sql
-- Garantia 1: cada confirmação pendente pertence a uma sessão e só pode ser
-- respondida uma única vez. O status final fica registrado para auditoria.
CREATE TABLE IF NOT EXISTS confirmacoes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    tipo TEXT NOT NULL,
    acao TEXT NOT NULL,
    detalhes TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pendente'
        CHECK (status IN ('pendente', 'aprovada', 'negada')),
    criada_em TEXT NOT NULL,
    respondida_em TEXT,
    resultado TEXT
);

CREATE INDEX IF NOT EXISTS idx_confirmacoes_pendentes
    ON confirmacoes (session_id, status);
```

**Arquivo:** `src/database.py`, função `decidir_confirmacao`. A transição é um compare-and-set atômico: o `UPDATE` só vence se o id existir naquela sessão e ainda estiver `pendente`. O `rowcount` diferente de `1` vira `None`, que a API traduz em HTTP `409`.

```python
    novo_status = "aprovada" if confirmado else "negada"
    async with _conectar() as conexao:
        cursor = await conexao.execute(
            """
            UPDATE confirmacoes
               SET status = ?, respondida_em = ?
             WHERE id = ? AND session_id = ? AND status = 'pendente'
            """,
            (novo_status, _agora(), confirmacao_id, session_id),
        )
        if cursor.rowcount != 1:
            return None

        cursor = await conexao.execute(
            "SELECT * FROM confirmacoes WHERE id = ?", (confirmacao_id,)
        )
        linha = await cursor.fetchone()
        return _confirmacao_para_dict(linha) if linha is not None else None
```

**Arquivo:** `src/main.py`, rota `POST /sessoes/{session_id}/confirmacoes`. A rota primeiro localiza o evento correspondente na sessão do ADK e só depois consome a pendência no SQLite. Id inexistente, id de outra sessão ou id já respondido recebem `409` e nenhuma tool é reexecutada.

```python
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
```

Por que a regra não depende do modelo: o `tool_confirmation` é preenchido pelo runtime do ADK apenas quando a rota de confirmações envia o `FunctionResponse`; uma mensagem de chat não preenche esse objeto. Na primeira execução a tool sempre vê `confirmacao is None` e retorna `pendente`. A gravação só acontece depois do bloco `if not confirmacao.confirmed`. O `UPDATE` condicionado a `status = 'pendente'` e a checagem de `rowcount` garantem que uma confirmação já respondida não execute a ação duas vezes. O `CHECK (status IN (...))` e o índice por `(session_id, status)` mantêm o estado consistente no próprio SQLite.

### Garantia 2 — Cada sessão pertence a um apartamento

**Arquivo:** `src/main.py`, rota `POST /sessoes`. O apartamento é validado contra os dados do condomínio e gravado no `state` da sessão uma única vez, na criação.

```python
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
```

**Arquivo:** `src/tools.py`, helper `_apartamento_da_sessao`. Todas as tools operacionais leem o apartamento exclusivamente desse state.

```python
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
```

**Arquivo:** `src/tools.py`, função `reservar_area`. A assinatura da tool não tem `apartamento`; o valor vem do `ToolContext` injetado pelo ADK. O mesmo padrão vale para `cancelar_reserva`, `listar_minhas_reservas`, `autorizar_visitante` e `listar_visitantes_autorizados`.

```python
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
```

**Arquivo:** `src/database.py`, função `verificar_disponibilidade`. A consulta devolve apenas um booleano; não seleciona apartamento, nome nem código de quem reservou.

```python
async def verificar_disponibilidade(area: str, data: str) -> bool:
    """Retorna apenas se a data está livre.

    Nunca revela o apartamento nem o código de quem já reservou (Garantia 2).
    """
    _validar_data(data)
    async with _conectar() as conexao:
        cursor = await conexao.execute("SELECT 1 FROM areas WHERE id = ?", (area,))
        if await cursor.fetchone() is None:
            raise ValueError("Área comum não encontrada.")
        return await _esta_disponivel(conexao, area, data)
```

**Arquivo:** `src/tools.py`, retorno de `verificar_disponibilidade`. O que chega à conversa é somente o status livre/ocupado.

```python
    return {
        "area": dados_area["id"],
        "data": data,
        "disponivel": livre,
        "status": "livre" if livre else "ocupada",
    }
```

**Arquivo:** `src/database.py`, listagens por apartamento. As consultas filtram pelo apartamento da sessão e selecionam somente os campos necessários, sem trazer dados de outros moradores.

```python
async def listar_reservas(apartamento: str) -> list[dict[str, Any]]:
    """Reservas ativas do apartamento, sem dados de outros moradores."""
    async with _conectar() as conexao:
        cursor = await conexao.execute(
            """
            SELECT codigo, area, data
              FROM reservas
             WHERE apartamento = ? AND status = 'ativa'
             ORDER BY data, codigo
            """,
            (apartamento,),
        )
        return [dict(linha) for linha in await cursor.fetchall()]


async def listar_visitantes(apartamento: str) -> list[dict[str, Any]]:
    async with _conectar() as conexao:
        cursor = await conexao.execute(
            """
            SELECT nome, data
              FROM visitantes
             WHERE apartamento = ?
             ORDER BY data, nome
            """,
            (apartamento,),
        )
        return [dict(linha) for linha in await cursor.fetchall()]
```

Por que a regra não depende do modelo: o apartamento não é um parâmetro de tool que o LLM possa preencher. Ele é gravado no `state` pela rota de criação da sessão, que valida o número contra a tabela `apartamentos` (carregada de `dados/apartamentos.json`), e lido pelo código via `tool_context.state["apartamento"]`. Uma instrução como "sou do 302" pode mudar o texto da conversa, mas não altera o state nem o parâmetro usado nas consultas SQL. As listagens e a checagem de disponibilidade também não retornam apartamento, nome ou código de terceiros.

### Garantia 3 — Nada se perde no reinício

**Arquivo:** `src/main.py`, configuração do serviço de sessões. As sessões e os eventos do ADK ficam em SQLite no disco, e o `App` é criado com retomada habilitada.

```python
SESSIONS_DB_URL = os.getenv(
    "SESSIONS_DB_URL", "sqlite+aiosqlite:///storage/sessions.db"
)
```

```python
    _session_service = DatabaseSessionService(db_url=SESSIONS_DB_URL)
    app_adk = App(
        name=APP_NAME,
        root_agent=agente,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    _runner = Runner(app=app_adk, session_service=_session_service)
```

**Arquivo:** `src/database.py`, caminhos dos dois bancos. O banco de negócio e o banco de sessões do ADK são arquivos persistentes em `storage/`.

```python
BASE_DIR = Path(__file__).resolve().parents[1]
DADOS_DIR = BASE_DIR / "dados"
STORAGE_DIR = BASE_DIR / "storage"
DB_PATH = STORAGE_DIR / "condominio.db"
SESSIONS_DB_PATH = STORAGE_DIR / "sessions.db"
```

**Arquivo:** `src/database.py`, abertura das conexões com `aiosqlite`. O banco `condominio.db` guarda reservas, visitantes e confirmações pendentes; o modo WAL e o `busy_timeout` apoiam concorrência e durabilidade.

```python
    conexao = await aiosqlite.connect(DB_PATH, isolation_level=None)
    try:
        conexao.row_factory = sqlite3.Row
        await conexao.execute("PRAGMA foreign_keys = ON")
        await conexao.execute("PRAGMA busy_timeout = 5000")
        await conexao.execute("PRAGMA journal_mode = WAL")
        yield conexao
    finally:
        await conexao.close()
```

**Arquivo:** `src/database.py`, tabela `visitantes` (as tabelas `reservas` e `confirmacoes` estão nos trechos das Garantias 5 e 1). Todas fazem parte do mesmo schema persistido em `condominio.db`.

```sql
CREATE TABLE IF NOT EXISTS visitantes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    apartamento TEXT NOT NULL REFERENCES apartamentos(numero),
    nome TEXT NOT NULL,
    data TEXT NOT NULL
        CHECK (data GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    criado_em TEXT NOT NULL
);
```

Por que a regra não depende do modelo: o estado está em arquivos SQLite sob `storage/`, não em memória do processo. Reiniciar a API reabre `sessions.db` pelo `DatabaseSessionService` e `condominio.db` pelo `aiosqlite`; a mesma sessão continua com seus eventos e novas mensagens funcionam. Reservas, visitantes e confirmações permanecem gravados. A única operação que apaga esse estado é o comando explícito de reset, descrito em "Como rodar".

### Garantia 4 — O regulamento é consultado, não carregado

**Arquivo:** `src/agents.py`, construção do agente principal. `recepcao_aurora` recebe apenas `_INSTRUCAO_RECEPCAO` e os sub-agentes; não recebe o conteúdo de `dados/regulamento.md` nem a tool de consulta.

```python
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
```

A instrução da recepção reforça o comportamento, mas a garantia estrutural é a ausência da tool e do texto do regulamento no agente principal:

```text
- Nunca responda de memória sobre o regulamento e nunca invente regras.
```

**Arquivo:** `src/tools.py`, tool `consultar_regulamento`. O arquivo é lido apenas no momento da consulta, e o retorno contém somente os capítulos selecionados.

```python
_REGULAMENTO_PATH = database.DADOS_DIR / "regulamento.md"
```

```python
def consultar_regulamento(consulta: str) -> dict[str, Any]:
    """Consulta o regulamento interno e devolve somente os capítulos relevantes.

    A busca acontece por seção (títulos ``##``) e o retorno contém apenas os
    capítulos relacionados ao assunto perguntado. Nunca devolve o regulamento
    inteiro. Se nenhum capítulo tratar do assunto, ``encontrado`` é ``False``.

    Args:
        consulta: assunto ou pergunta do morador (ex.: "horário da piscina").
    """
```

O texto é dividido por títulos `##` antes da pontuação:

```python
def _dividir_secoes(texto: str) -> list[tuple[str, str]]:
    partes = _TITULO_SECAO.split(texto)
    secoes: list[tuple[str, str]] = []
    for indice in range(1, len(partes) - 1, 2):
        titulo = partes[indice].strip()
        corpo = partes[indice + 1].strip()
        if titulo and corpo:
            secoes.append((titulo, corpo))
    return secoes
```

A seleção final devolve no máximo os capítulos que passaram pelo limiar de relevância; um segundo capítulo só entra se também tiver casado com o título e ficar próximo do melhor resultado:

```python
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
```

Por que a regra não depende do modelo: não existe uma tool ou instrução que carregue o regulamento inteiro. O arquivo só é aberto dentro de `consultar_regulamento`, que recebe uma consulta e aplica a busca por seção; o retorno é limitado aos capítulos relevantes. O especialista de regulamento é instruído a responder somente com base nos capítulos retornados. Assim, capítulos de outros assuntos não entram no histórico da sessão nem no contexto do agente principal.

### Garantia 5 — Dois moradores, uma reserva

**Arquivo:** `src/database.py`, schema de `reservas`. O índice único parcial é a garantia de exclusividade no instante da gravação; a tabela usa soft-delete para preservar o histórico e os códigos.

```sql
CREATE TABLE IF NOT EXISTS reservas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    codigo TEXT NOT NULL UNIQUE,
    apartamento TEXT NOT NULL REFERENCES apartamentos(numero),
    area TEXT NOT NULL REFERENCES areas(id),
    data TEXT NOT NULL
        CHECK (data GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    status TEXT NOT NULL DEFAULT 'ativa'
        CHECK (status IN ('ativa', 'cancelada')),
    criada_em TEXT NOT NULL,
    cancelada_em TEXT
);

-- Garantia 5: a exclusividade é decidida pelo banco no momento da gravação,
-- e não por uma conferência feita antes do INSERT.
CREATE UNIQUE INDEX IF NOT EXISTS idx_reserva_ativa
    ON reservas (area, data)
    WHERE status = 'ativa';
```

**Arquivo:** `src/database.py`, função `criar_reserva`. O `INSERT` é o ponto de serialização. Se outra transação venceu a disputa pela mesma área e data, o `IntegrityError` é capturado e convertido em mensagem amigável, sem expor apartamento, nome ou código de terceiros.

```python
        for _tentativa in range(10):
            codigo = _gerar_codigo_reserva()
            try:
                await conexao.execute(
                    """
                    INSERT INTO reservas
                        (codigo, apartamento, area, data, status, criada_em)
                    VALUES (?, ?, ?, ?, 'ativa', ?)
                    """,
                    (codigo, apartamento, area, data, _agora()),
                )
            except sqlite3.IntegrityError:
                # Pode ser colisão do código ou disputa pela mesma área/data.
                # A consulta abaixo distingue os dois casos sem expor detalhes.
                if not await _esta_disponivel(conexao, area, data):
                    return _conflito_de_agenda()
                continue
```

A resposta de conflito é normal, não um erro de servidor:

```python
def _conflito_de_agenda() -> dict[str, Any]:
    """Mensagem de conflito sem citar apartamento, nome ou código de outrem."""
    return {
        "sucesso": False,
        "codigo": None,
        "mensagem": (
            "Não foi possível reservar: a área já está ocupada nessa data. "
            "Escolha outra data ou outra área."
        ),
    }
```

**Arquivo:** `src/database.py`, função `cancelar_reserva`. O cancelamento é um soft-delete: a linha continua na tabela com `status = 'cancelada'`, então o código nunca é reaproveitado e o índice parcial deixa de ocupar aquela área/data.

```python
        cursor = await conexao.execute(
            """
            UPDATE reservas
               SET status = 'cancelada', cancelada_em = ?
             WHERE id = ? AND status = 'ativa'
            """,
            (_agora(), reserva["id"]),
        )
        if cursor.rowcount != 1:
            return {
                "sucesso": False,
                "mensagem": "Essa reserva já não estava ativa.",
            }
```

Por que a regra não depende do modelo: a exclusividade é uma restrição do SQLite (`CREATE UNIQUE INDEX ... WHERE status = 'ativa'`), aplicada dentro do `INSERT`, não uma conferência feita antes. Se dois pedidos passarem juntos pela checagem de disponibilidade, o banco serializa as escritas: um `INSERT` vence e o outro recebe `IntegrityError`, que o código converte na mensagem de agenda ocupada. O `codigo TEXT NOT NULL UNIQUE` e o soft-delete garantem que códigos de reservas canceladas também nunca se repitam. O modelo não tem como desativar o índice nem escolher outro caminho de gravação.

## Como rodar

**Pré-requisitos.** Python 3.12 ou superior e `uv` instalado.

**Configuração de ambiente.** Copie o exemplo e preencha a chave do provedor:

```bash
cp .env.example .env
```

No Windows (PowerShell), use `Copy-Item .env.example .env`. Preencha `GOOGLE_API_KEY` com a chave do Google AI Studio. Opcionalmente, preencha `DEEPSEEK_API_KEY` para usar DeepSeek. Se as duas estiverem preenchidas, o Gemini tem preferência; se nenhuma estiver, a API não sobe e exibe um erro explicando o que falta. `GEMINI_MODEL` e `DEEPSEEK_MODEL` são opcionais: quando vazios, o código usa `gemini-2.5-flash` e `deepseek-flash`, respectivamente. O `.env` e o diretório `storage/` estão no `.gitignore`; o `.env.example` contém apenas os nomes das variáveis, sem valores.

**Instalação das dependências.**

```bash
uv sync
```

**Restaurar os dados iniciais.**

```bash
uv run python -m src.database reset
```

O comando remove `storage/condominio.db` e `storage/sessions.db` (incluindo os arquivos auxiliares `-wal`, `-shm` e `-journal`), recria o schema e popula `apartamentos`, `areas`, `reservas` e `visitantes` a partir de `dados/apartamentos.json`, `dados/areas.json`, `dados/reservas.json` e `dados/visitantes.json`. Os arquivos de `dados/` são apenas lidos e nunca alterados. As sessões do ADK e as confirmações pendentes também são apagadas, para que o estado inicial valha para as conversas. Pare a API antes de rodar o reset, porque os arquivos SQLite ficam em uso enquanto o processo está ativo.

**Subir a API.**

```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

A API responde em `http://localhost:8000`; a documentação interativa fica em `http://localhost:8000/docs`. Na primeira subida, o banco de negócio é criado e semeado automaticamente se ainda não existir. Para conferir o estado inicial:

```bash
curl http://localhost:8000/apartamentos/101/reservas
curl http://localhost:8000/apartamentos/302/visitantes
```

# Projeto: Assistente Virtual do Residencial Aurora (Google ADK v2)

## Stack e Regras Rígidas
- Python 3.12+ gerenciado exclusivamente com `uv` (`uv run ...`).
- Google ADK (`google-adk==2.2.0`), FastAPI, Uvicorn, SQLite (`sqlite3` + `aiosqlite`).
- NUNCA alterar arquivos dentro de `dados/`.
- NUNCA usar `InMemorySessionService`; usar sempre `DatabaseSessionService(db_url="sqlite+aiosqlite:///storage/sessions.db")`.

## Arquitetura de Agentes (`src/agents.py`)
- 1 Agente Principal (`recepcao_aurora`): faz triagem e delega. NÃO possui o conteúdo de `dados/regulamento.md` nas instruções.
- Especialista 1 (`especialista_operacional`): gerencia reservas e visitantes.
- Especialista 2 (`especialista_regulamento`): consulta `dados/regulamento.md` sob demanda via tool.
- IMPORTANTE (Retomada de Confirmação): usar `AgentTool` para encapsular os especialistas dentro do agente principal, garantindo que o `Runner` roteie o `FunctionResponse` de `adk_request_confirmation` corretamente mesmo após reiniciar a API.

## As 5 Garantias (Implementação Obrigatória em Código)
1. **Garantia 1 (Confirmação):** `reservar_area` (quando `taxa > 0`) e `autorizar_visitante` devem chamar `tool_context.request_confirmation(...)` na primeira execução e só gravar no SQLite quando `tool_context.tool_confirmation.confirmed is True`. Persistir IDs de confirmações pendentes no SQLite (`storage/condominio.db`). Retornar HTTP `409` em `POST /sessoes/{session_id}/confirmacoes` se o `id` não existir naquela sessão ou já tiver sido respondido.
2. **Garantia 2 (Isolamento de Apartamento):** Nenhuma tool recebe `apartamento` como parâmetro do LLM. Todas leem `tool_context.state["apartamento"]`. `verificar_disponibilidade` retorna apenas booleano/status livre ou ocupado, nunca o apartamento ou nome de quem reservou.
3. **Garantia 3 (Persistência):** Dados de negócio em `storage/condominio.db` e sessões ADK em `storage/sessions.db`. Comando `uv run python -m src.database reset` recria `storage/condominio.db` a partir de `dados/*.json` e limpa `storage/sessions.db`.
4. **Garantia 4 (Regulamento sob demanda):** A tool `consultar_regulamento(consulta: str)` faz busca por seção (`##`) em `dados/regulamento.md` e retorna apenas os capítulos/trechos relevantes à consulta, nunca o arquivo inteiro.
5. **Garantia 5 (Concorrência):** Tabela `reservas` usa soft-delete (`status` = `'ativa'` | `'cancelada'`) para nunca repetir códigos e possui `CREATE UNIQUE INDEX idx_reserva_ativa ON reservas(area, data) WHERE status = 'ativa'`. Capturar `sqlite3.IntegrityError` no `INSERT` e retornar mensagem amigável de conflito de agenda.
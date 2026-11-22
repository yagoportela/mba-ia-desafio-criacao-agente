"""Persistência de dados do Assistente Virtual do Residencial Aurora.

O banco de negócio fica em ``storage/condominio.db`` (SQLite) e as sessões do
Google ADK em ``storage/sessions.db``. Este módulo concentra o esquema, a carga
inicial a partir de ``dados/*.json`` e as operações usadas pelas tools.

Garantias implementadas aqui:

* **Garantia 1** — a tabela ``confirmacoes`` guarda cada pedido de confirmação
  como ``pendente`` até a resposta chegar pela rota da API. A transição é
  atômica, então uma confirmação já respondida nunca é executada de novo.
* **Garantia 3** — reservas, visitantes e confirmações ficam em disco e
  sobrevivem ao reinício da API.
* **Garantia 5** — o índice único parcial ``idx_reserva_ativa`` garante, no
  instante do ``INSERT``, que existe no máximo uma reserva ativa por área e
  data. Reservas canceladas permanecem na tabela (soft-delete) e não ocupam o
  índice.

Comandos disponíveis::

    uv run python -m src.database reset   # recria tudo a partir de dados/
    uv run python -m src.database init    # cria o banco se ainda não existir
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import secrets
import sqlite3
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

BASE_DIR = Path(__file__).resolve().parents[1]
DADOS_DIR = BASE_DIR / "dados"
STORAGE_DIR = BASE_DIR / "storage"
DB_PATH = STORAGE_DIR / "condominio.db"
SESSIONS_DB_PATH = STORAGE_DIR / "sessions.db"

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS meta (
    chave TEXT PRIMARY KEY,
    valor TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS apartamentos (
    numero TEXT PRIMARY KEY,
    morador TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS areas (
    id TEXT PRIMARY KEY,
    nome TEXT NOT NULL,
    taxa REAL NOT NULL CHECK (taxa >= 0)
);

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

CREATE TABLE IF NOT EXISTS visitantes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    apartamento TEXT NOT NULL REFERENCES apartamentos(numero),
    nome TEXT NOT NULL,
    data TEXT NOT NULL
        CHECK (data GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    criado_em TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_visitante_unico
    ON visitantes (apartamento, nome, data);

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
"""


# ---------------------------------------------------------------------------
# Infraestrutura
# ---------------------------------------------------------------------------


def _agora() -> str:
    """Data/hora UTC em ISO 8601, usada para auditoria das gravações."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validar_data(data: str) -> str:
    """Garante o formato AAAA-MM-DD antes de qualquer gravação."""
    if not isinstance(data, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", data) is None:
        raise ValueError("Data inválida. Use o formato AAAA-MM-DD.")
    try:
        date.fromisoformat(data)
    except ValueError as exc:
        raise ValueError("Data inválida. Use o formato AAAA-MM-DD.") from exc
    return data


@asynccontextmanager
async def _conectar() -> AsyncIterator[aiosqlite.Connection]:
    """Abre uma conexão curta com o banco de negócio.

    ``isolation_level = None`` deixa o controle de transações explícito; cada
    comando de escrita é atômico por si só, e operações compostas usam
    ``BEGIN IMMEDIATE`` quando precisam de serialização.
    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    conexao = await aiosqlite.connect(DB_PATH, isolation_level=None)
    try:
        conexao.row_factory = sqlite3.Row
        await conexao.execute("PRAGMA foreign_keys = ON")
        await conexao.execute("PRAGMA busy_timeout = 5000")
        await conexao.execute("PRAGMA journal_mode = WAL")
        yield conexao
    finally:
        await conexao.close()


async def _aplicar_esquema(conexao: aiosqlite.Connection) -> None:
    await conexao.executescript(_ESQUEMA)


def _ler_json(nome_arquivo: str) -> list[dict[str, Any]]:
    caminho = DADOS_DIR / nome_arquivo
    with caminho.open(encoding="utf-8") as arquivo:
        conteudo = json.load(arquivo)
    if not isinstance(conteudo, list):
        raise ValueError(f"Formato inesperado em {caminho}: era esperada uma lista.")
    return conteudo


async def _semear(conexao: aiosqlite.Connection) -> None:
    """Popula o banco com o estado inicial de ``dados/*.json``.

    Deve ser chamado dentro de uma transação: o marcador em ``meta`` só é
    gravado se toda a carga der certo.
    """
    apartamentos = _ler_json("apartamentos.json")
    areas = _ler_json("areas.json")
    reservas = _ler_json("reservas.json")
    visitantes = _ler_json("visitantes.json")
    agora = _agora()

    await conexao.executemany(
        "INSERT INTO apartamentos (numero, morador) VALUES (?, ?)",
        [(item["numero"], item["morador"]) for item in apartamentos],
    )
    await conexao.executemany(
        "INSERT INTO areas (id, nome, taxa) VALUES (?, ?, ?)",
        [(item["id"], item["nome"], float(item["taxa"])) for item in areas],
    )
    await conexao.executemany(
        """
        INSERT INTO reservas (codigo, apartamento, area, data, status, criada_em)
        VALUES (?, ?, ?, ?, 'ativa', ?)
        """,
        [
            (item["codigo"], item["apartamento"], item["area"], item["data"], agora)
            for item in reservas
        ],
    )
    await conexao.executemany(
        "INSERT INTO visitantes (apartamento, nome, data, criado_em) VALUES (?, ?, ?, ?)",
        [
            (item["apartamento"], item["nome"], item["data"], agora)
            for item in visitantes
        ],
    )
    await conexao.execute(
        "INSERT INTO meta (chave, valor) VALUES ('inicializado_em', ?)",
        (agora,),
    )


async def inicializar_banco() -> None:
    """Cria o esquema e, na primeira execução, carrega os dados iniciais.

    É idempotente e seguro para chamar na inicialização da API: um banco já
    inicializado não é semeado de novo.
    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    async with _conectar() as conexao:
        await _aplicar_esquema(conexao)
        cursor = await conexao.execute(
            "SELECT 1 FROM meta WHERE chave = 'inicializado_em'"
        )
        if await cursor.fetchone() is not None:
            return

        # BEGIN IMMEDIATE serializa a primeira carga entre processos concorrentes.
        await conexao.execute("BEGIN IMMEDIATE")
        try:
            cursor = await conexao.execute(
                "SELECT 1 FROM meta WHERE chave = 'inicializado_em'"
            )
            if await cursor.fetchone() is None:
                await _semear(conexao)
            await conexao.commit()
        except Exception:
            await conexao.rollback()
            raise


# ---------------------------------------------------------------------------
# Consultas usadas pelas tools
# ---------------------------------------------------------------------------


async def obter_apartamento(numero: str) -> dict[str, Any] | None:
    async with _conectar() as conexao:
        cursor = await conexao.execute(
            "SELECT numero, morador FROM apartamentos WHERE numero = ?", (numero,)
        )
        linha = await cursor.fetchone()
        return dict(linha) if linha is not None else None


async def listar_areas() -> list[dict[str, Any]]:
    async with _conectar() as conexao:
        cursor = await conexao.execute(
            "SELECT id, nome, taxa FROM areas ORDER BY nome"
        )
        return [dict(linha) for linha in await cursor.fetchall()]


async def obter_area(area_id: str) -> dict[str, Any] | None:
    async with _conectar() as conexao:
        cursor = await conexao.execute(
            "SELECT id, nome, taxa FROM areas WHERE id = ?", (area_id,)
        )
        linha = await cursor.fetchone()
        return dict(linha) if linha is not None else None


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


async def _esta_disponivel(
    conexao: aiosqlite.Connection, area: str, data: str
) -> bool:
    cursor = await conexao.execute(
        """
        SELECT 1
          FROM reservas
         WHERE area = ? AND data = ? AND status = 'ativa'
         LIMIT 1
        """,
        (area, data),
    )
    return await cursor.fetchone() is None


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


# ---------------------------------------------------------------------------
# Escritas de negócio
# ---------------------------------------------------------------------------


def _gerar_codigo_reserva() -> str:
    """Código novo em formato livre; a unicidade é garantida pelo UNIQUE."""
    return f"RSV-{secrets.randbelow(900_000) + 100_000}"


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


async def criar_reserva(
    apartamento: str, area: str, data: str
) -> dict[str, Any]:
    """Grava uma reserva ativa para o apartamento da sessão.

    A exclusividade é do banco: mesmo que dois pedidos passem juntos pela
    conferência, o índice ``idx_reserva_ativa`` deixa apenas um INSERT vencer.
    """
    try:
        _validar_data(data)
    except ValueError as exc:
        return {"sucesso": False, "codigo": None, "mensagem": str(exc)}

    async with _conectar() as conexao:
        cursor = await conexao.execute(
            "SELECT nome, taxa FROM areas WHERE id = ?", (area,)
        )
        dados_area = await cursor.fetchone()
        if dados_area is None:
            return {
                "sucesso": False,
                "codigo": None,
                "mensagem": "Área comum não encontrada.",
            }

        cursor = await conexao.execute(
            "SELECT 1 FROM apartamentos WHERE numero = ?", (apartamento,)
        )
        if await cursor.fetchone() is None:
            return {
                "sucesso": False,
                "codigo": None,
                "mensagem": "Apartamento não encontrado.",
            }

        # Caminho rápido para a mensagem amigável. A garantia real é o índice.
        if not await _esta_disponivel(conexao, area, data):
            return _conflito_de_agenda()

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

            taxa = float(dados_area["taxa"])
            mensagem = (
                f"Reserva {codigo} confirmada para {dados_area['nome']} em {data}."
            )
            if taxa > 0:
                valor = f"{taxa:.2f}".replace(".", ",")
                mensagem += f" A reserva gerou cobrança de R$ {valor}."
            return {
                "sucesso": True,
                "codigo": codigo,
                "area": area,
                "data": data,
                "taxa": taxa,
                "cobranca": taxa > 0,
                "mensagem": mensagem,
            }

    return {
        "sucesso": False,
        "codigo": None,
        "mensagem": (
            "Não foi possível concluir a reserva agora. Tente novamente em instantes."
        ),
    }


async def cancelar_reserva(
    apartamento: str,
    *,
    codigo: str | None = None,
    area: str | None = None,
    data: str | None = None,
) -> dict[str, Any]:
    """Cancela uma reserva ativa do próprio apartamento (soft-delete).

    ``codigo`` nunca é reaproveitado, porque a linha continua na tabela.
    """
    if not codigo and not (area and data):
        return {
            "sucesso": False,
            "mensagem": "Informe o código da reserva ou a área e a data.",
        }

    if data is not None:
        try:
            _validar_data(data)
        except ValueError as exc:
            return {"sucesso": False, "mensagem": str(exc)}

    async with _conectar() as conexao:
        if codigo:
            cursor = await conexao.execute(
                """
                SELECT id, codigo, area, data
                  FROM reservas
                 WHERE codigo = ? AND apartamento = ? AND status = 'ativa'
                """,
                (codigo.strip().upper(), apartamento),
            )
        else:
            cursor = await conexao.execute(
                """
                SELECT id, codigo, area, data
                  FROM reservas
                 WHERE apartamento = ? AND area = ? AND data = ?
                   AND status = 'ativa'
                """,
                (apartamento, area, data),
            )

        reserva = await cursor.fetchone()
        if reserva is None:
            return {
                "sucesso": False,
                "mensagem": "Não encontrei uma reserva ativa sua com esses dados.",
            }

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

        return {
            "sucesso": True,
            "codigo": reserva["codigo"],
            "area": reserva["area"],
            "data": reserva["data"],
            "mensagem": f"Reserva {reserva['codigo']} cancelada com sucesso.",
        }


async def autorizar_visitante(
    apartamento: str, nome: str, data: str
) -> dict[str, Any]:
    """Registra a autorização de entrada de um visitante."""
    nome = nome.strip()
    if not nome:
        return {"sucesso": False, "mensagem": "Informe o nome do visitante."}

    try:
        _validar_data(data)
    except ValueError as exc:
        return {"sucesso": False, "mensagem": str(exc)}

    async with _conectar() as conexao:
        cursor = await conexao.execute(
            "SELECT 1 FROM apartamentos WHERE numero = ?", (apartamento,)
        )
        if await cursor.fetchone() is None:
            return {"sucesso": False, "mensagem": "Apartamento não encontrado."}

        try:
            await conexao.execute(
                """
                INSERT INTO visitantes (apartamento, nome, data, criado_em)
                VALUES (?, ?, ?, ?)
                """,
                (apartamento, nome, data, _agora()),
            )
        except sqlite3.IntegrityError:
            return {
                "sucesso": True,
                "nome": nome,
                "data": data,
                "mensagem": f"{nome} já estava autorizado(a) para {data}.",
            }

        return {
            "sucesso": True,
            "nome": nome,
            "data": data,
            "mensagem": f"Entrada de {nome} autorizada para {data}.",
        }


# ---------------------------------------------------------------------------
# Confirmações pendentes (Garantia 1)
# ---------------------------------------------------------------------------


def _confirmacao_para_dict(linha: sqlite3.Row) -> dict[str, Any]:
    registro = dict(linha)
    try:
        registro["detalhes"] = json.loads(registro["detalhes"])
    except (TypeError, json.JSONDecodeError):
        registro["detalhes"] = {}
    if registro.get("resultado"):
        try:
            registro["resultado"] = json.loads(registro["resultado"])
        except (TypeError, json.JSONDecodeError):
            pass
    return registro


async def registrar_confirmacao(
    *,
    session_id: str,
    confirmacao_id: str,
    tipo: str,
    acao: str,
    detalhes: Mapping[str, Any],
) -> None:
    """Persiste uma confirmação pendente antes de interromper a execução."""
    async with _conectar() as conexao:
        await conexao.execute(
            """
            INSERT INTO confirmacoes
                (id, session_id, tipo, acao, detalhes, status, criada_em)
            VALUES (?, ?, ?, ?, ?, 'pendente', ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (
                confirmacao_id,
                session_id,
                tipo,
                acao,
                json.dumps(detalhes, ensure_ascii=False),
                _agora(),
            ),
        )


async def listar_confirmacoes_pendentes(session_id: str) -> list[dict[str, Any]]:
    """Confirmações que ainda podem ser respondidas na sessão."""
    async with _conectar() as conexao:
        cursor = await conexao.execute(
            """
            SELECT id, acao, detalhes
              FROM confirmacoes
             WHERE session_id = ? AND status = 'pendente'
             ORDER BY criada_em, rowid
            """,
            (session_id,),
        )
        return [_confirmacao_para_dict(linha) for linha in await cursor.fetchall()]


async def obter_confirmacao(
    session_id: str, confirmacao_id: str
) -> dict[str, Any] | None:
    async with _conectar() as conexao:
        cursor = await conexao.execute(
            "SELECT * FROM confirmacoes WHERE id = ? AND session_id = ?",
            (confirmacao_id, session_id),
        )
        linha = await cursor.fetchone()
        return _confirmacao_para_dict(linha) if linha is not None else None


async def decidir_confirmacao(
    *, session_id: str, confirmacao_id: str, confirmado: bool
) -> dict[str, Any] | None:
    """Responde uma confirmação de forma atômica.

    Retorna ``None`` quando o id não existe naquela sessão ou quando já foi
    respondido — condição que a API traduz em HTTP 409.
    """
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


async def registrar_resultado_confirmacao(
    confirmacao_id: str, resultado: Mapping[str, Any]
) -> None:
    """Guarda o desfecho da ação executada, útil para auditoria."""
    async with _conectar() as conexao:
        await conexao.execute(
            "UPDATE confirmacoes SET resultado = ? WHERE id = ?",
            (json.dumps(resultado, ensure_ascii=False), confirmacao_id),
        )


# ---------------------------------------------------------------------------
# Reset e CLI
# ---------------------------------------------------------------------------


def _remover_arquivo_sqlite(caminho: Path) -> None:
    # O banco de negócio usa WAL (-wal/-shm); as sessões do ADK podem usar o
    # rollback journal padrão (-journal). Todos precisam sair junto.
    for sufixo in ("", "-wal", "-shm", "-journal"):
        alvo = Path(f"{caminho}{sufixo}")
        if not alvo.exists():
            continue
        try:
            alvo.unlink()
        except PermissionError as exc:
            raise RuntimeError(
                f"Não foi possível remover {alvo}. Pare a API antes de rodar o reset."
            ) from exc


async def _contar_registros() -> dict[str, int]:
    tabelas = ("apartamentos", "areas", "reservas", "visitantes")
    async with _conectar() as conexao:
        contagem: dict[str, int] = {}
        for tabela in tabelas:
            cursor = await conexao.execute(f"SELECT COUNT(*) AS total FROM {tabela}")
            linha = await cursor.fetchone()
            contagem[tabela] = int(linha["total"])
        return contagem


async def resetar_banco() -> dict[str, int]:
    """Recria ``condominio.db`` a partir de ``dados/`` e apaga as sessões.

    As sessões são apagadas para que o estado inicial também valha para as
    conversas do ADK, sem confirmações pendentes órfãs.
    """
    _remover_arquivo_sqlite(DB_PATH)
    _remover_arquivo_sqlite(SESSIONS_DB_PATH)
    await inicializar_banco()
    return await _contar_registros()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.database",
        description="Gerencia o banco de dados do Residencial Aurora.",
    )
    subcomandos = parser.add_subparsers(dest="comando", required=True)
    subcomandos.add_parser(
        "reset",
        help=(
            "Recria storage/condominio.db a partir de dados/ e apaga "
            "storage/sessions.db."
        ),
    )
    subcomandos.add_parser(
        "init",
        help="Cria o banco e carrega os dados iniciais apenas se ainda não existirem.",
    )
    argumentos = parser.parse_args(argv)

    try:
        if argumentos.comando == "reset":
            contagem = asyncio.run(resetar_banco())
            print("Banco recriado a partir de dados/.")
            print(f"  storage/condominio.db: {DB_PATH}")
            print(f"  storage/sessions.db:   apagado ({SESSIONS_DB_PATH})")
            print(
                "  registros: "
                + ", ".join(f"{tabela}={total}" for tabela, total in contagem.items())
            )
        else:
            asyncio.run(inicializar_banco())
            print(f"Banco inicializado em {DB_PATH}.")
    except (RuntimeError, ValueError, sqlite3.Error) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

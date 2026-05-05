"""
db.py — SQLite 数据库持久化层 (MemCore v0.3)

为 MemoryCore 提供底层数据库操作，使用 Python 标准库 sqlite3，
不引入任何第三方依赖。

表结构：
    memories (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        query             TEXT NOT NULL,
        content           TEXT NOT NULL,
        fingerprint       TEXT NOT NULL,
        query_vector      BLOB,
        created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        access_count      INTEGER DEFAULT 0,
        protection_score  REAL DEFAULT 1.0
    )
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _try_add_column(conn: sqlite3.Connection, col_name: str, col_type: str, table: str = "memories") -> None:
    """尝试为旧数据库表添加新列，若列已存在则静默忽略。

    Parameters
    ----------
    conn : sqlite3.Connection
        数据库连接。
    col_name : str
        列名。
    col_type : str
        列类型定义（如 "INTEGER DEFAULT 0"）。
    table : str, default "memories"
        表名。
    """
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
    except sqlite3.OperationalError:
        pass


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    """初始化 SQLite 数据库，创建 memories 表（如不存在）。

    Parameters
    ----------
    db_path : str
        数据库文件路径。":memory:" 表示纯内存数据库，适用于测试。

    Returns
    -------
    sqlite3.Connection
        数据库连接对象，已设置 row_factory = sqlite3.Row。
    """
    # check_same_thread=False: 允许跨线程使用（FastAPI 异步端点 + 测试注入均需要）
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            query             TEXT NOT NULL,
            content           TEXT NOT NULL,
            fingerprint       TEXT NOT NULL,
            query_vector      BLOB,
            keyword_vector    BLOB,
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            access_count      INTEGER DEFAULT 0,
            protection_score  REAL DEFAULT 1.0
        )
        """
    )
    conn.commit()
    # 兼容旧数据库（没有新字段的旧表）
    _try_add_column(conn, "access_count", "INTEGER DEFAULT 0")
    _try_add_column(conn, "protection_score", "REAL DEFAULT 1.0")
    _try_add_column(conn, "keyword_vector", "BLOB")
    _try_add_column(conn, "successor_id", "INTEGER")
    _try_add_column(conn, "uuid", "TEXT")
    return conn


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def insert_memory(
    conn: sqlite3.Connection,
    query: str,
    content: str,
    fingerprint: str,
    query_vector: bytes | None = None,
    keyword_vector: bytes | None = None,
    uuid_str: str | None = None,
) -> int:
    """插入一条新的记忆记录，返回新记录的自增 ID。

    Parameters
    ----------
    conn : sqlite3.Connection
        数据库连接。
    query : str
        标准化查询主题。
    content : str
        核心事实陈述。
    fingerprint : str
        空格分隔的关键词串。
    query_vector : bytes | None, default None
        查询文本的嵌入向量序列化字节。
    keyword_vector : bytes | None, default None
        关键词向量的 L2 归一化序列化字节（128维）。
    uuid_str : str | None, default None
        外部 UUID。None 时由调用方生成。

    Returns
    -------
    int
        新插入记录的 lastrowid。
    """
    cursor = conn.execute(
        "INSERT INTO memories (query, content, fingerprint, query_vector, keyword_vector, uuid) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (query, content, fingerprint, query_vector, keyword_vector, uuid_str),
    )
    conn.commit()
    return cursor.lastrowid


def delete_oldest_memory(conn: sqlite3.Connection) -> None:
    """删除 id 最小的记忆记录（即最早的记忆），实现 FIFO 遗忘。

    Parameters
    ----------
    conn : sqlite3.Connection
        数据库连接。
    """
    conn.execute("DELETE FROM memories WHERE id = (SELECT MIN(id) FROM memories)")
    conn.commit()


def delete_memory_by_id(conn: sqlite3.Connection, memory_id: int) -> None:
    """按 id 删除指定的记忆记录。

    Parameters
    ----------
    conn : sqlite3.Connection
        数据库连接。
    memory_id : int
        要删除的记忆记录的 id。
    """
    conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.commit()


def get_all_memories(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """获取所有记忆记录，按 id 升序排列。

    Parameters
    ----------
    conn : sqlite3.Connection
        数据库连接。

    Returns
    -------
    list[dict]
        每条记录包含 id, query, content, fingerprint, query_vector,
        created_at, access_count, protection_score 八个字段。
    """
    rows = conn.execute(
        "SELECT id, uuid, query, content, fingerprint, query_vector, keyword_vector, "
        "successor_id, created_at, access_count, protection_score "
        "FROM memories ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


def update_memory_stats(
    conn: sqlite3.Connection,
    memory_id: int,
    access_count: int,
    protection_score: float,
    auto_commit: bool = True,
) -> None:
    """更新指定记忆的 access_count 和 protection_score。

    Parameters
    ----------
    conn : sqlite3.Connection
        数据库连接。
    memory_id : int
        要更新的记忆记录的 id。
    access_count : int
        新的访问计数值。
    protection_score : float
        新的保护分数值。
    auto_commit : bool, default True
        是否自动提交。批量调用时设为 False，最后手动 commit。
    """
    conn.execute(
        "UPDATE memories SET access_count = ?, protection_score = ? WHERE id = ?",
        (access_count, protection_score, memory_id),
    )
    if auto_commit:
        conn.commit()


def batch_update_memory_stats(
    conn: sqlite3.Connection,
    updates: list[tuple[int, int, float]],
) -> None:
    """批量更新多条记忆的 access_count 和 protection_score。

    单次事务提交，减少 I/O 开销。

    Parameters
    ----------
    conn : sqlite3.Connection
        数据库连接。
    updates : list[tuple[int, int, float]]
        每个元组为 (memory_id, access_count, protection_score)。
    """
    conn.executemany(
        "UPDATE memories SET access_count = ?, protection_score = ? WHERE id = ?",
        [(ac, ps, mid) for mid, ac, ps in updates],
    )
    conn.commit()


def update_memory_content(
    conn: sqlite3.Connection,
    memory_id: int,
    query: str,
    content: str,
    fingerprint: str,
    query_vector: bytes | None = None,
    keyword_vector: bytes | None = None,
) -> None:
    """更新指定记忆的内容字段（用于写入去重时合并）。

    Parameters
    ----------
    conn : sqlite3.Connection
        数据库连接。
    memory_id : int
        要更新的记忆记录的 id。
    query : str
        新的查询主题。
    content : str
        新的核心事实（保留更长版本）。
    fingerprint : str
        合并后的关键词串。
    query_vector : bytes | None, default None
        重新编码后的向量（可选）。
    keyword_vector : bytes | None, default None
        重新编码后的关键词向量（可选）。
    """
    conn.execute(
        """UPDATE memories
           SET query = ?, content = ?, fingerprint = ?,
               query_vector = COALESCE(?, query_vector),
               keyword_vector = COALESCE(?, keyword_vector),
               access_count = access_count + 1,
               protection_score = MIN(protection_score + 0.2, ?)
           WHERE id = ?""",
        (query, content, fingerprint, query_vector, keyword_vector,
         10.0, memory_id),
    )
    conn.commit()


def update_memory_successor(
    conn: sqlite3.Connection,
    memory_id: int,
    successor_id: int,
) -> None:
    """原子更新记忆的后继者 ID（事务保护）。

    Parameters
    ----------
    conn : sqlite3.Connection
        数据库连接。
    memory_id : int
        要更新的原记忆 ID。
    successor_id : int
        后继记忆 ID。
    """
    with conn:
        conn.execute(
            "UPDATE memories SET successor_id = ? WHERE id = ?",
            (successor_id, memory_id),
        )


def get_count(conn: sqlite3.Connection) -> int:
    """返回记忆表中的记录总数。

    Parameters
    ----------
    conn : sqlite3.Connection
        数据库连接。

    Returns
    -------
    int
        记录总数。
    """
    row = conn.execute("SELECT COUNT(*) AS cnt FROM memories").fetchone()
    return row["cnt"]


# ===========================================================================
# 事件表（v0.6 事件索引）
# ===========================================================================

EVENT_DB_FILENAME = "events.db"


def init_event_db(db_dir: str) -> sqlite3.Connection:
    """初始化事件数据库，创建 events 和 event_fact_sources 表。

    Parameters
    ----------
    db_dir : str
        数据库文件存放目录。

    Returns
    -------
    sqlite3.Connection
        事件数据库连接。
    """
    if db_dir == ":memory:":
        conn = sqlite3.connect(":memory:", check_same_thread=False)
    else:
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, EVENT_DB_FILENAME)
        conn = sqlite3.connect(db_path, check_same_thread=False)

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id        TEXT UNIQUE NOT NULL,
            theme           TEXT NOT NULL,
            timestamp       REAL NOT NULL,
            status          TEXT DEFAULT 'closed',
            facts_json      TEXT NOT NULL DEFAULT '[]',
            vector_blob     BLOB,
            fingerprint     TEXT DEFAULT '',
            access_count    INTEGER DEFAULT 0,
            last_accessed   REAL DEFAULT 0.0,
            last_updated    REAL DEFAULT 0.0,
            parent_event_id TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_timestamp
        ON events(timestamp)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_status
        ON events(status)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_event_id
        ON events(event_id)
    """)
    conn.commit()
    # 兼容旧数据库（没有新字段的旧表）
    _try_add_column(conn, "last_updated", "REAL DEFAULT 0.0", table="events")
    _try_add_column(conn, "parent_event_id", "TEXT", table="events")
    _try_add_column(conn, "doc_mode", "INTEGER DEFAULT 0", table="events")
    conn.commit()
    return conn


def insert_event(
    conn: sqlite3.Connection,
    event_id: str,
    theme: str,
    timestamp: float,
    status: str,
    facts_json: str,
    vector_blob: bytes | None = None,
    fingerprint: str = "",
    access_count: int = 0,
    last_accessed: float = 0.0,
    parent_event_id: str | None = None,
    doc_mode: bool = False,
) -> int:
    """插入一条事件记录，返回自增 ID。

    Parameters
    ----------
    conn : sqlite3.Connection
        事件数据库连接。
    event_id : str
        全局唯一事件 ID。
    theme : str
        事件主题。
    timestamp : float
        time.time() 时间戳。
    status : str
        "open" 或 "closed"。
    facts_json : str
        JSON 数组字符串（事件内知识列表）。
    vector_blob : bytes | None, default None
        BGE 事件向量序列化。
    fingerprint : str, default ''
        合并后的关键词串。
    access_count : int, default 0
        访问计数。
    last_accessed : float, default 0.0
        最后访问时间。
    parent_event_id : str | None, default None
        父事件 ID，用于事件链。
    doc_mode : bool, default False
        文档模式标记。True 表示整份文档存储，不精炼不切分。

    Returns
    -------
    int
        新记录的 id。
    """
    conn.execute("BEGIN")
    try:
        cursor = conn.execute(
            """INSERT INTO events
               (event_id, theme, timestamp, status, facts_json,
             vector_blob, fingerprint, access_count, last_accessed,
             last_updated, parent_event_id, doc_mode)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_id, theme, timestamp, status, facts_json,
             vector_blob, fingerprint, access_count, last_accessed,
             timestamp, parent_event_id, doc_mode),
        )
        conn.commit()
        return cursor.lastrowid
    except Exception:
        conn.rollback()
        raise


def get_all_events(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """返回所有事件记录，按时间倒序。"""
    return conn.execute(
        "SELECT * FROM events ORDER BY timestamp DESC"
    ).fetchall()


def get_event_by_id(conn: sqlite3.Connection, event_id: str) -> sqlite3.Row | None:
    """按 event_id 查询事件。"""
    return conn.execute(
        "SELECT * FROM events WHERE event_id = ?", (event_id,)
    ).fetchone()


def update_event_facts(
    conn: sqlite3.Connection,
    event_id: str,
    facts_json: str,
    fingerprint: str,
    vector_blob: bytes | None = None,
    last_updated: float | None = None,
) -> None:
    """更新事件的知识列表、关键词和向量（事务保护）。"""
    conn.execute("BEGIN")
    try:
        if vector_blob is not None and last_updated is not None:
            conn.execute(
                """UPDATE events
                   SET facts_json = ?, fingerprint = ?, vector_blob = ?,
                       last_updated = ?
                   WHERE event_id = ?""",
                (facts_json, fingerprint, vector_blob, last_updated, event_id),
            )
        elif vector_blob is not None:
            conn.execute(
                """UPDATE events
                   SET facts_json = ?, fingerprint = ?, vector_blob = ?
                   WHERE event_id = ?""",
                (facts_json, fingerprint, vector_blob, event_id),
            )
        elif last_updated is not None:
            conn.execute(
                """UPDATE events
                   SET facts_json = ?, fingerprint = ?, last_updated = ?
                   WHERE event_id = ?""",
                (facts_json, fingerprint, last_updated, event_id),
            )
        else:
            conn.execute(
                """UPDATE events
                   SET facts_json = ?, fingerprint = ?
                   WHERE event_id = ?""",
                (facts_json, fingerprint, event_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def update_event_status(
    conn: sqlite3.Connection,
    event_id: str,
    status: str,
) -> None:
    """更新事件状态（open/closed，事务保护）。"""
    conn.execute("BEGIN")
    try:
        conn.execute(
            "UPDATE events SET status = ? WHERE event_id = ?",
            (status, event_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def update_event_access(
    conn: sqlite3.Connection,
    event_id: str,
    access_count: int,
    last_accessed: float,
) -> None:
    """更新事件访问计数（事务保护）。"""
    conn.execute("BEGIN")
    try:
        conn.execute(
            "UPDATE events SET access_count = ?, last_accessed = ? WHERE event_id = ?",
            (access_count, last_accessed, event_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def delete_event(
    conn: sqlite3.Connection,
    event_id: str,
) -> None:
    """删除事件记录（事务保护）。"""
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM events WHERE event_id = ?", (event_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise

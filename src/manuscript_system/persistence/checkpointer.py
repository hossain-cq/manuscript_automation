from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def build_checkpointer(db_path: str) -> SqliteSaver:
    """Build a SqliteSaver from a raw connection rather than
    `SqliteSaver.from_conn_string(...)`.

    `from_conn_string` is a context manager in current langgraph-checkpoint-sqlite
    releases, so calling it directly (as the original prototype did) hands back
    a context-manager object rather than a saver. Constructing from an explicit
    `sqlite3.Connection` is the documented, version-stable path.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)

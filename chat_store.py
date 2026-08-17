"""
Local SQLite persistence for chat sessions, so chat history survives an app
restart (not just a page refresh). No user accounts — this is a single-instance
tool, so all chats are visible to whoever opens the app, same as a local desktop
app rather than a multi-tenant product.
"""
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "chat_history.db"


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            chat_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_trace TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats (chat_id)
        )
    """)
    conn.commit()
    return conn


def create_chat(title="New chat"):
    conn = _connect()
    chat_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO chats (chat_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (chat_id, title, now, now),
    )
    conn.commit()
    conn.close()
    return chat_id


def list_chats():
    """Returns chats newest-first: [{'chat_id', 'title', 'updated_at'}, ...]"""
    conn = _connect()
    rows = conn.execute(
        "SELECT chat_id, title, updated_at FROM chats ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [{"chat_id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]


def get_messages(chat_id):
    """Returns [{'role', 'content', 'tool_trace'}, ...] in chronological order."""
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content, tool_trace FROM messages WHERE chat_id = ? ORDER BY id ASC",
        (chat_id,),
    ).fetchall()
    conn.close()
    out = []
    for role, content, tool_trace in rows:
        out.append({
            "role": role,
            "content": content,
            "tool_trace": json.loads(tool_trace) if tool_trace else None,
        })
    return out


def add_message(chat_id, role, content, tool_trace=None):
    conn = _connect()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO messages (chat_id, role, content, tool_trace, created_at) VALUES (?, ?, ?, ?, ?)",
        (chat_id, role, content, json.dumps(tool_trace) if tool_trace else None, now),
    )
    conn.execute("UPDATE chats SET updated_at = ? WHERE chat_id = ?", (now, chat_id))
    conn.commit()
    conn.close()


def rename_chat_from_first_message(chat_id, first_user_message):
    """Auto-titles a chat from its first user message, so the sidebar list is
    actually readable instead of showing 'New chat' for every entry."""
    title = first_user_message.strip().replace("\n", " ")
    if len(title) > 45:
        title = title[:42] + "..."
    conn = _connect()
    conn.execute("UPDATE chats SET title = ? WHERE chat_id = ?", (title, chat_id))
    conn.commit()
    conn.close()


def delete_chat(chat_id):
    conn = _connect()
    conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()

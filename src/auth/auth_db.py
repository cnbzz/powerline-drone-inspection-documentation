"""Auth persistence — stdlib ``sqlite3``, separate DB from the fault store.

Two tables in ``outputs/db/auth.db``:

    users(id, username UNIQUE, password_hash, salt, role, is_active,
          created_at, last_login)
    activity_log(id, username, action, detail, timestamp)

Design notes:
    - Soft delete only: deactivate via ``is_active=0``; rows are never removed.
    - Plain-text passwords are never accepted into storage or logs — callers pass
      a password, we hash it here.
    - On first init the default admin (admin / admin123) is seeded and a console
      warning is printed.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from src.auth.security import generate_salt, hash_password, verify_password

# Default DB location — mirrors faults.db sibling but is a SEPARATE file.
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = _ROOT / "outputs" / "db" / "auth.db"

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"


def _now() -> str:
    """ISO-8601 timestamp string (used for created_at/last_login/log)."""
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def _connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Yield a sqlite3 connection with row access by name. Commits on success."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create tables if missing and seed the default admin on first run.

    Idempotent: safe to call on every app start.
    """
    fresh_admin = False
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt          TEXT NOT NULL,
                role          TEXT NOT NULL DEFAULT 'user',
                is_active     INTEGER NOT NULL DEFAULT 1,
                created_at    TEXT NOT NULL,
                last_login    TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                username  TEXT,
                action    TEXT NOT NULL,
                detail    TEXT,
                timestamp TEXT NOT NULL
            )
            """
        )
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM users WHERE username = ?",
            (DEFAULT_ADMIN_USERNAME,),
        ).fetchone()
        if row["c"] == 0:
            salt = generate_salt()
            conn.execute(
                """
                INSERT INTO users
                    (username, password_hash, salt, role, is_active, created_at)
                VALUES (?, ?, ?, 'admin', 1, ?)
                """,
                (
                    DEFAULT_ADMIN_USERNAME,
                    hash_password(DEFAULT_ADMIN_PASSWORD, salt),
                    salt,
                    _now(),
                ),
            )
            fresh_admin = True

    if fresh_admin:
        # Console warning only — never print the password hash or plaintext beyond
        # this one-time default-credential reminder.
        print(
            "[AUTH] Varsayilan admin olusturuldu (admin / admin123). "
            "Varsayilan admin sifresini ilk giriste degistirin."
        )


def create_user(
    username: str,
    password: str,
    role: str = "user",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> bool:
    """Create a new user. Returns ``True`` on success, ``False`` if the username
    already exists. The password is hashed here; plaintext never persists."""
    username = username.strip()
    if not username or not password:
        return False
    salt = generate_salt()
    try:
        with _connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO users
                    (username, password_hash, salt, role, is_active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
                """,
                (username, hash_password(password, salt), salt, role, _now()),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def get_user(
    username: str, db_path: str | Path = DEFAULT_DB_PATH
) -> Optional[dict]:
    """Return the user row as a dict, or ``None`` if not found."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def verify_user(
    username: str, password: str, db_path: str | Path = DEFAULT_DB_PATH
) -> Optional[dict]:
    """Validate credentials.

    Returns the user dict (without secret fields) on success, else ``None``.
    Inactive users are rejected. On success ``last_login`` is updated.
    """
    user = get_user(username, db_path)
    if user is None:
        return None
    if not user["is_active"]:
        return None
    if not verify_password(password, user["salt"], user["password_hash"]):
        return None

    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE users SET last_login = ? WHERE username = ?",
            (_now(), username),
        )
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "is_active": bool(user["is_active"]),
        "last_login": user["last_login"],
    }


def change_password(
    username: str,
    new_password: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    old_password: Optional[str] = None,
) -> bool:
    """Set a new password for ``username``.

    If ``old_password`` is given it must match the current one (self-service
    change). Pass ``old_password=None`` for an admin reset. Returns ``True`` on
    success.
    """
    if not new_password:
        return False
    user = get_user(username, db_path)
    if user is None:
        return False
    if old_password is not None and not verify_password(
        old_password, user["salt"], user["password_hash"]
    ):
        return False

    salt = generate_salt()
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE username = ?",
            (hash_password(new_password, salt), salt, username),
        )
    return True


def set_active(
    username: str, is_active: bool, db_path: str | Path = DEFAULT_DB_PATH
) -> bool:
    """Soft enable/disable a user. Never deletes. Returns ``True`` on success."""
    with _connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE users SET is_active = ? WHERE username = ?",
            (1 if is_active else 0, username),
        )
        return cur.rowcount > 0


def list_users(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    """Return all users (no password material) ordered by creation time."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, username, role, is_active, created_at, last_login
            FROM users ORDER BY created_at ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def log_activity(
    username: Optional[str],
    action: str,
    detail: str = "",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Append an entry to the activity log. Never store plaintext passwords here."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO activity_log (username, action, detail, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (username, action, detail, _now()),
        )


def get_activity_log(
    db_path: str | Path = DEFAULT_DB_PATH,
    username: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 500,
) -> list[dict]:
    """Return activity-log rows, newest first, optionally filtered."""
    clauses: list[str] = []
    params: list[object] = []
    if username:
        clauses.append("username = ?")
        params.append(username)
    if action:
        clauses.append("action = ?")
        params.append(action)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT id, username, action, detail, timestamp
            FROM activity_log {where}
            ORDER BY id DESC LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(r) for r in rows]

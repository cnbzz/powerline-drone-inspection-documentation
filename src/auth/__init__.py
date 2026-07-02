"""Authentication & authorization for the HHATS dashboard.

Self-contained module: a *separate* SQLite database (``outputs/db/auth.db``) holds
users and an activity log. It never touches the fault database (``faults.db``).

Public surface:
    - :mod:`security`  — password hashing (PBKDF2-HMAC, stdlib only).
    - :mod:`auth_db`   — user/activity persistence (stdlib ``sqlite3``).
    - :mod:`session`   — Streamlit session helpers (login gate, admin gate, logout).
"""

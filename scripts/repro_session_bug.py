"""Repro: session loss on page reload = the 'kicked back to login' bug.

A raw HTML `<a href="?fault_detail=ID" target="_self">` (fault-row click, nav
cards, KPI cards) performs a FULL-PAGE navigation. Streamlit does NOT persist
st.session_state across a full page load — a fresh session starts, so
`st.session_state["auth_user"]` is gone and require_login() re-shows the login
form.

AppTest models this exactly: every AppTest.from_file(...).run() is a FRESH
session (empty session_state) — i.e. what a browser reload produces. So:
  * fresh session (no auth_user)  -> login form  (== the reported bug)
  * same page WITH auth_user      -> renders normally (session preserved case)
This isolates the cause to session loss, not to the guard logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

FAULTS = "src/dashboard/pages/3_Faults.py"
FAKE_USER = {"id": 99, "username": "saha1", "role": "user",
             "is_active": True, "last_login": "2026-07-05T10:00:00"}


def _has_login_form(at: AppTest) -> bool:
    """Detect the login gate by its unique content, not by widget counts
    (the authenticated Faults page also has many text inputs)."""
    for md in at.markdown:
        if "HHATS Giriş" in (md.value or ""):
            return True
    # Fallback: a form submit button labelled exactly "Giriş".
    for b in at.button:
        if (getattr(b, "label", "") or "").strip() == "Giriş":
            return True
    return False


def main() -> None:
    print("=" * 68)
    print("REPRO — session loss on reload == kicked to login")
    print("=" * 68)

    # 1) FRESH session (what a full-page reload gives) — no auth_user.
    at = AppTest.from_file(str(_ROOT / FAULTS), default_timeout=60)
    at.run()
    bug = _has_login_form(at) and not at.exception
    print(f"[1] Fresh session (=reload), no auth_user  -> login form shown: "
          f"{_has_login_form(at)}  (exception={at.exception})")
    print(f"    => reproduces the bug: {'YES' if bug else 'NO'}")

    # 2) SAME page but session preserved (auth_user present) — no login form.
    at2 = AppTest.from_file(str(_ROOT / FAULTS), default_timeout=60)
    at2.session_state["auth_user"] = FAKE_USER
    at2.run()
    ok = (not _has_login_form(at2)) and not at2.exception
    print(f"[2] Session preserved (auth_user set)      -> login form shown: "
          f"{_has_login_form(at2)}  (exception={at2.exception})")
    print(f"    => works when session survives: {'YES' if ok else 'NO'}")

    # 3) Confirm the reload trigger: fault rows are wrapped in reload anchors.
    src = (Path(_ROOT) / FAULTS).read_text(encoding="utf-8")
    n_anchor = src.count('href="?fault_detail=')
    n_close = src.count('href="?close_detail=')
    print(f"[3] Fault-row reload anchors in source ('?fault_detail='): {n_anchor}")
    print(f"    close-detail reload anchor ('?close_detail='): {n_close}")

    print("=" * 68)
    verdict = bug and ok and n_anchor > 0
    print(f"ROOT CAUSE CONFIRMED: {'YES' if verdict else 'NO'}")
    print("  Fresh session (reload) -> no auth_user -> login gate. Fault-row")
    print("  clicks use full-reload <a href> anchors -> fresh session each click.")


if __name__ == "__main__":
    main()

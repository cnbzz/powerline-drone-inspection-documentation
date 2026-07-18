"""Verify the reload-persistence fix: a fresh session (= what a full-page reload
gives) with a VALID session cookie restores the login instead of showing the
login gate. Also verifies security is preserved (no/invalid cookie -> login).

We can't set a real browser cookie in AppTest, so we monkeypatch
``session._get_cookie_token`` (the single cookie-read seam) to emulate the
cookie the browser would send after a reload.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.auth import auth_db, session  # noqa: E402

FAULTS = "src/dashboard/pages/3_Faults.py"


def _has_login_form(at: AppTest) -> bool:
    for md in at.markdown:
        if "HHATS Giriş" in (md.value or ""):
            return True
    for b in at.button:
        if (getattr(b, "label", "") or "").strip() == "Giriş":
            return True
    return False


def _run(token_value):
    """Run Faults in a FRESH session (empty session_state) with the cookie-read
    seam returning ``token_value`` (emulating the post-reload request cookie)."""
    orig = session._get_cookie_token
    session._get_cookie_token = lambda: token_value
    try:
        at = AppTest.from_file(str(_ROOT / FAULTS), default_timeout=60)
        at.run()
        return at
    finally:
        session._get_cookie_token = orig


def main() -> None:
    auth_db.init_db()
    # A real, valid server-side token for the seeded admin user.
    valid = auth_db.create_session("admin")

    print("=" * 68)
    print("FIX VERIFY — reload recovery via session cookie")
    print("=" * 68)

    # 1) Reload WITH a valid cookie -> restored, NO login gate. (the fix)
    at = _run(valid)
    fixed = (not _has_login_form(at)) and not at.exception
    restored = at.session_state["auth_user"] if "auth_user" in at.session_state else None
    print(f"[1] reload + valid cookie -> login_form={_has_login_form(at)} "
          f"exception={bool(at.exception)}  restored_user="
          f"{restored.get('username') if restored else None}")
    print(f"    => stays logged in after reload: {'YES' if fixed else 'NO'}")

    # 2) Reload with NO cookie -> login gate (security preserved).
    at2 = _run(None)
    gated_none = _has_login_form(at2) and not at2.exception
    print(f"[2] reload + no cookie    -> login_form={_has_login_form(at2)}  "
          f"=> gated: {'YES' if gated_none else 'NO'}")

    # 3) Reload with an INVALID/expired cookie -> login gate (no bypass).
    at3 = _run("garbage-token-xyz")
    gated_bad = _has_login_form(at3) and not at3.exception
    print(f"[3] reload + bad cookie   -> login_form={_has_login_form(at3)}  "
          f"=> gated: {'YES' if gated_bad else 'NO'}")

    # 4) After logout (token deleted server-side) the same cookie no longer works.
    auth_db.delete_session(valid)
    at4 = _run(valid)
    gated_after_logout = _has_login_form(at4) and not at4.exception
    print(f"[4] deleted-token cookie  -> login_form={_has_login_form(at4)}  "
          f"=> gated after logout: {'YES' if gated_after_logout else 'NO'}")

    print("=" * 68)
    ok = fixed and gated_none and gated_bad and gated_after_logout
    print(f"RESULT: {'ALL PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

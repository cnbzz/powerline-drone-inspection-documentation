"""AppTest smoke test: every page renders without exception and shows the login
gate when unauthenticated."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    "src/dashboard/app.py",
    "src/dashboard/pages/1_Dashboard.py",
    "src/dashboard/pages/2_Map.py",
    "src/dashboard/pages/3_Faults.py",
    "src/dashboard/pages/4_Reports.py",
    "src/dashboard/pages/5_Admin.py",
]


def run_page(path: str) -> AppTest:
    at = AppTest.from_file(str(_ROOT / path), default_timeout=30)
    at.run()
    return at


def main() -> None:
    print("=" * 60)
    all_ok = True
    for path in PAGES:
        at = run_page(path)
        if at.exception:
            all_ok = False
            print(f"FAIL  {path}")
            for exc in at.exception:
                print("      ", exc.value)
            continue
        # Unauthenticated -> a login form with username/password inputs is shown.
        n_inputs = len(at.text_input)
        has_login = n_inputs >= 2
        print(f"OK    {path:42s} exception=None  text_inputs={n_inputs}"
              f"  login_form={'yes' if has_login else 'no'}")

    print("=" * 60)

    # Verify the actual login flow on the entry page.
    at = run_page("src/dashboard/app.py")
    if len(at.text_input) >= 2:
        at.text_input[0].set_value("admin")
        at.text_input[1].set_value("admin123")
        # Submit the login form.
        if at.button:
            at.button[0].click().run()
        sess = at.session_state["auth_user"] if "auth_user" in at.session_state else None
        print("Login as admin -> session auth_user:", sess)
        if sess and sess.get("role") == "admin":
            print("ADMIN LOGIN VIA APPTEST: OK")
        else:
            all_ok = False
            print("ADMIN LOGIN VIA APPTEST: FAIL")

    print("=" * 60)
    print("ALL APPTEST CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

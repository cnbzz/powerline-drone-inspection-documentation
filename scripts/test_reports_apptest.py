"""AppTest for Reports A1: authenticated render + interactive preview/select flow.

Verifies (in a real Streamlit session):
  * page renders with no exception,
  * clicking a row's 👁️ preview button sets session_state['rep_preview'] and
    survives the rerun (no exception),
  * toggling a row checkbox records the fault into session_state['rep_selected'].
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

REPORTS = "src/dashboard/pages/4_Reports.py"
ADMIN = {"id": 1, "username": "admin", "role": "admin",
         "is_active": True, "last_login": "2026-07-06T10:00:00"}


def _auth_run() -> AppTest:
    at = AppTest.from_file(str(_ROOT / REPORTS), default_timeout=90)
    at.session_state["auth_user"] = ADMIN  # bypass login gate (session preserved)
    at.run()
    return at


def main() -> None:
    at = _auth_run()
    print(f"[1] authenticated render -> exception={bool(at.exception)}")
    if at.exception:
        for e in at.exception:
            print("   ", e.value)
        sys.exit(1)

    prev_btns = [b for b in at.button if b.key and b.key.startswith("rep_prev_")
                 and b.key != "rep_prev_close"]
    checkboxes = [c for c in at.checkbox if c.key and c.key.startswith("rep_cb_")]
    print(f"    preview buttons={len(prev_btns)}  row checkboxes={len(checkboxes)}")
    assert prev_btns and checkboxes, "interactive widgets missing"

    # Click a preview button -> selection persists across the rerun.
    target_fid = prev_btns[0].key.replace("rep_prev_", "")
    prev_btns[0].click().run()
    got = at.session_state["rep_preview"] if "rep_preview" in at.session_state else None
    print(f"[2] clicked preview for {target_fid} -> rep_preview={got} "
          f"exception={bool(at.exception)}")
    assert got == target_fid and not at.exception, "preview did not persist"

    # A PDF preview iframe (base64 data-URI) should now be embedded.
    has_iframe = any("data:application/pdf;base64," in (m.value or "")
                     for m in at.markdown)
    print(f"[3] base64 PDF preview embedded: {has_iframe}")

    # Toggle a checkbox -> fault recorded in the selection set.
    cb = [c for c in at.checkbox if c.key and c.key.startswith("rep_cb_")][0]
    cb_fid = cb.key.replace("rep_cb_", "")
    cb.check().run()
    sel = at.session_state["rep_selected"] if "rep_selected" in at.session_state else set()
    print(f"[4] checked {cb_fid} -> in rep_selected={cb_fid in sel} "
          f"exception={bool(at.exception)}")
    assert cb_fid in sel and not at.exception, "checkbox selection not recorded"

    print("=" * 60)
    print("REPORTS APPTEST: ALL PASS")


if __name__ == "__main__":
    main()

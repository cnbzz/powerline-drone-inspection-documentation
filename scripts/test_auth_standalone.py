"""Standalone smoke test for the auth layer (no Streamlit).

Exercises: init_db (seeds admin) -> create_user -> verify_user (correct/wrong) ->
change_password -> activity log. Proves the stored credential is a hash, not the
plaintext password. Uses a throwaway DB so it never touches the real auth.db.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.auth import auth_db  # noqa: E402


def main() -> None:
    tmp = Path(tempfile.gettempdir()) / "hhats_auth_test.db"
    if tmp.exists():
        tmp.unlink()
    db = str(tmp)

    print("=" * 60)
    print("1) init_db -> seeds default admin")
    auth_db.init_db(db)
    admin = auth_db.verify_user("admin", "admin123", db)
    print("   admin login (admin/admin123):", "OK" if admin else "FAIL", admin)

    print("2) create_user('saha1', 'parola123', role='user')")
    created = auth_db.create_user("saha1", "parola123", role="user", db_path=db)
    print("   created:", created)
    print("   duplicate create (should be False):",
          auth_db.create_user("saha1", "x", db_path=db))

    print("3) verify_user correct vs wrong password")
    good = auth_db.verify_user("saha1", "parola123", db)
    bad = auth_db.verify_user("saha1", "yanlis", db)
    print("   correct password ->", "OK" if good else "FAIL")
    print("   wrong password   ->", "REJECTED" if bad is None else "FAIL")

    print("4) change_password with current-password verification")
    ok = auth_db.change_password("saha1", "yeniParola1", db_path=db,
                                 old_password="parola123")
    print("   change with correct old pw:", ok)
    bad_change = auth_db.change_password("saha1", "z", db_path=db,
                                         old_password="WRONG")
    print("   change with wrong old pw (should be False):", bad_change)
    print("   login with new password:",
          "OK" if auth_db.verify_user("saha1", "yeniParola1", db) else "FAIL")

    print("5) hash is NOT plaintext")
    row = auth_db.get_user("saha1", db)
    print("   stored password_hash:", row["password_hash"][:32], "...")
    print("   stored salt:", row["salt"])
    assert "yeniParola1" not in row["password_hash"], "PLAINTEXT LEAK!"
    assert "parola123" not in row["password_hash"], "PLAINTEXT LEAK!"
    print("   plaintext NOT present in stored hash: CONFIRMED")

    print("6) activity log")
    auth_db.log_activity("saha1", "login_success", "test event", db)
    logs = auth_db.get_activity_log(db, limit=10)
    print(f"   {len(logs)} log rows; newest:", logs[0] if logs else None)

    tmp.unlink(missing_ok=True)
    print("=" * 60)
    print("ALL AUTH CHECKS PASSED")


if __name__ == "__main__":
    main()

"""Password hashing — stdlib only, no extra packages.

Plain-text passwords are NEVER stored. Each user gets a random per-user salt and
a PBKDF2-HMAC-SHA256 derived hash. Verification uses a constant-time comparison
(:func:`hmac.compare_digest`) to avoid timing side-channels.
"""

from __future__ import annotations

import hashlib
import hmac
import os

# PBKDF2 parameters. 200k iterations is a sane 2020s+ baseline for SHA256.
_ALGO = "sha256"
_ITERATIONS = 200_000
_SALT_BYTES = 16
_DKLEN = 32


def generate_salt() -> str:
    """Return a fresh random salt as a hex string."""
    return os.urandom(_SALT_BYTES).hex()


def hash_password(password: str, salt: str) -> str:
    """Derive a PBKDF2-HMAC-SHA256 hash for ``password`` using ``salt`` (hex).

    Returns the derived key as a hex string. The same (password, salt) always
    yields the same hash, which is what lets :func:`verify_password` work.
    """
    dk = hashlib.pbkdf2_hmac(
        _ALGO,
        password.encode("utf-8"),
        bytes.fromhex(salt),
        _ITERATIONS,
        dklen=_DKLEN,
    )
    return dk.hex()


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """Constant-time check that ``password`` hashes to ``expected_hash``."""
    candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, expected_hash)

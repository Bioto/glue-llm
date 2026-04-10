"""Stable opaque identifiers derived from API keys.

Used only for in-process rate-limit bucketing and log-safe key labels. This is
**not** a stored password verifier: we use HMAC-BLAKE2s with a fixed
domain-separation key (appropriate for keyed fingerprints). BLAKE2 avoids
SHA-256-based queries that static analyzers misclassify as password hashing.
"""

from __future__ import annotations

import hashlib
import hmac

# Domain-separation only (public constant). Prevents cross-protocol reuse; not a user password.
_HMAC_KEY = b"GlueLLM.api_key.fingerprint.v1"


def api_key_hmac_fingerprint(api_key: str) -> str:
    """Return a deterministic hex fingerprint for ``api_key`` (HMAC-BLAKE2s, 256-bit digest)."""
    return hmac.new(_HMAC_KEY, api_key.encode("utf-8"), hashlib.blake2s).hexdigest()

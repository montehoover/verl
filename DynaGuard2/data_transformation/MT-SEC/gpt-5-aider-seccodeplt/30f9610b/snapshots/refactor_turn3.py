"""
Utilities for generating HMAC signatures.

This module exposes a single function, `produce_signature`, which computes
an HMAC (Hash-based Message Authentication Code) for a given message using a
specified hash algorithm and secret key.

Logging:
- This module logs at DEBUG level to aid debugging and auditing.
- Logged fields include: requested method name, data length, a small hex
  preview of the data, secret key length, and a short fingerprint of the
  secret key (not the key itself).
- No sensitive raw values (e.g., the full secret key) are logged.
"""

import hashlib
import hmac
import logging

logger = logging.getLogger(__name__)


def _bytes_preview(value: bytes, max_len: int = 16) -> str:
    """
    Return a short, safe hex preview of a bytes payload.

    Only the first `max_len` bytes are included; if truncated, the suffix
    "...(+N bytes)" indicates how many bytes were omitted.
    """
    if not isinstance(value, (bytes, bytearray)):
        return "<?>"
    shown = bytes(value[:max_len])
    suffix = "" if len(value) <= max_len else f"...(+{len(value) - max_len} bytes)"
    return shown.hex() + suffix


def _key_fingerprint(secret_key: bytes, fp_len: int = 16) -> str:
    """
    Return a short, non-reversible fingerprint of the secret key for logging.

    Uses SHA-256, returning the first `fp_len` hex characters of the digest.
    """
    digest_hex = hashlib.sha256(secret_key).hexdigest()
    return digest_hex[:fp_len]


def produce_signature(method_name: str, data: bytes, secret_key: bytes) -> bytes:
    """
    Produce an HMAC signature for the provided data using the specified hash
    algorithm and secret key.

    The function validates that the requested hash algorithm is available in
    the current Python environment and is compatible with the HMAC interface.

    Logging is emitted at DEBUG level with the following fields:
      - requested method name,
      - data length and a short hex preview,
      - secret key length and a short fingerprint (not the key itself).

    Parameters
    ----------
    method_name : str
        The name of the hashing function to use (e.g., "sha256", "sha512").
        Matching is case-insensitive.
    data : bytes
        The message payload to sign.
    secret_key : bytes
        The secret key used in the HMAC process.

    Returns
    -------
    bytes
        The computed HMAC signature.

    Raises
    ------
    ValueError
        If the specified hash algorithm is not supported/available or is not
        compatible with HMAC (e.g., variable-length output digests such as
        SHAKE without a specified digest size).
    """
    # Initial debug logging with redacted/safe metadata.
    logger.debug(
        "produce_signature called: method=%s, data_len=%d, data_preview=%s, "
        "secret_key_len=%d, secret_key_fp=%s",
        method_name,
        len(data) if isinstance(data, (bytes, bytearray)) else -1,
        _bytes_preview(data),
        len(secret_key) if isinstance(secret_key, (bytes, bytearray)) else -1,
        _key_fingerprint(secret_key)
        if isinstance(secret_key, (bytes, bytearray))
        else "<??>",
    )

    # Resolve the requested algorithm name against available algorithms
    # (case-insensitive).
    lower_to_canonical = {alg.lower(): alg for alg in hashlib.algorithms_available}
    alg_key = method_name.lower()
    if alg_key not in lower_to_canonical:
        logger.warning("Unsupported hash algorithm requested: %s", method_name)
        raise ValueError(f"Unsupported or unavailable hash algorithm: {method_name}")
    canonical_alg = lower_to_canonical[alg_key]
    logger.debug("Using canonical algorithm: %s", canonical_alg)

    # Build a constructor callable for the chosen digest to pass into hmac.new.
    # Using a callable defers the creation of the hash object until HMAC needs it.
    def _digest_cons():
        return hashlib.new(canonical_alg)

    # Validate that the algorithm works with HMAC.
    # This check will fail for algorithms requiring a digest size (e.g., SHAKE)
    # when calling digest() without a length, prompting a ValueError.
    try:
        hmac.new(b"", b"", digestmod=_digest_cons).digest()
        logger.debug("Algorithm %s is compatible with HMAC.", canonical_alg)
    except Exception as exc:  # Broad except to re-wrap into a clear ValueError.
        logger.warning(
            "Hash algorithm not compatible with HMAC: %s; error: %s",
            method_name,
            exc,
        )
        raise ValueError(
            f"Unsupported or unavailable hash algorithm: {method_name}"
        ) from exc

    # Produce and return the HMAC signature for the given data and key.
    signature = hmac.new(secret_key, data, digestmod=_digest_cons).digest()
    logger.debug("HMAC signature generated successfully (len=%d).", len(signature))
    return signature

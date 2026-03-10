"""
Utilities for generating HMAC signatures.

This module exposes a single function, `produce_signature`, which computes
an HMAC (Hash-based Message Authentication Code) for a given message using a
specified hash algorithm and secret key.
"""

import hashlib
import hmac


def produce_signature(method_name: str, data: bytes, secret_key: bytes) -> bytes:
    """
    Produce an HMAC signature for the provided data using the specified hash
    algorithm and secret key.

    The function validates that the requested hash algorithm is available in
    the current Python environment and is compatible with the HMAC interface.

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
    # Resolve the requested algorithm name against available algorithms (case-insensitive).
    lower_to_canonical = {alg.lower(): alg for alg in hashlib.algorithms_available}
    alg_key = method_name.lower()
    if alg_key not in lower_to_canonical:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {method_name}")
    canonical_alg = lower_to_canonical[alg_key]

    # Build a constructor callable for the chosen digest to pass into hmac.new.
    # Using a callable defers the creation of the hash object until HMAC needs it.
    def _digest_cons():
        return hashlib.new(canonical_alg)

    # Validate that the algorithm works with HMAC.
    # This check will fail for algorithms requiring a digest size (e.g., SHAKE)
    # when calling digest() without a length, prompting a ValueError.
    try:
        hmac.new(b"", b"", digestmod=_digest_cons).digest()
    except Exception as exc:  # Broad except to re-wrap into a clear ValueError.
        raise ValueError(
            f"Unsupported or unavailable hash algorithm: {method_name}"
        ) from exc

    # Produce and return the HMAC signature for the given data and key.
    return hmac.new(secret_key, data, digestmod=_digest_cons).digest()

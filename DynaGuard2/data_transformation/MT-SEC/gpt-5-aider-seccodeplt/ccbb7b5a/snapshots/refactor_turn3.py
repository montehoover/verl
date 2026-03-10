"""
Hash utilities.

This module exposes a single function, compute_digest, that computes a
cryptographic digest of the provided content using a requested algorithm.
To promote security, it rejects weak algorithms (e.g., MD5, SHA-1) and
algorithms that require an explicit digest length (SHAKE variants), which
are not supported by this interface.

Logging:
    The module uses the standard logging library. A logger named after this
    module (``__name__``) is used to record:
      - attempts to compute a digest (algorithm and content length),
      - rejections due to unsafe or unavailable algorithms,
      - successful computations (algorithm and digest length),
      - unexpected errors (with stack traces).

    Library code should not configure logging. Configure handlers/levels in
    the application that imports this module.
"""

import hashlib
import logging

# Module-level logger for this utility.
logger = logging.getLogger(__name__)

# Algorithms that are considered weak or unsafe for general cryptographic use.
# These are explicitly rejected even if present in the local OpenSSL build.
UNSAFE_ALGORITHMS = {"md5", "sha1", "md5-sha1"}


def compute_digest(algorithm_name: str, content: bytes) -> bytes:
    """
    Compute the digest of the given content using the specified hash algorithm.

    Arguments:
        algorithm_name: The name of the hash algorithm to use. The value is
            case-insensitive (e.g., "SHA256", "sha256").
        content: The input data to hash as a bytes-like object.

    Returns:
        The computed hash as raw bytes (not a hexadecimal string).

    Raises:
        ValueError: If the algorithm name is not a string, the algorithm is
            unavailable on this system, or the algorithm is unsafe or
            unsupported by this interface.

    Notes:
        - Weak algorithms (MD5, SHA-1, and MD5-SHA1) are explicitly rejected.
        - SHAKE algorithms (XOFs) require a digest length and are therefore
          unsupported by this fixed-length digest interface.
    """
    # Log the invocation (avoid logging actual content; only its length).
    try:
        content_len = len(content)  # bytes should support len()
    except Exception:
        content_len = -1  # fallback if content has no length
    logger.info(
        "compute_digest called: algorithm=%s, content_len=%s",
        algorithm_name,
        content_len,
    )

    # Validate the algorithm name type early for clearer error messages.
    if not isinstance(algorithm_name, str):
        logger.error(
            "Algorithm name must be a string; got type=%s",
            type(algorithm_name).__name__,
        )
        raise ValueError("Algorithm name must be a string.")

    # Normalize the algorithm name to a common case for comparison.
    alg = algorithm_name.lower()

    # Reject weak/unsafe algorithms even if available in hashlib.
    if alg in UNSAFE_ALGORITHMS:
        logger.warning(
            "Blocked usage of unsafe hash algorithm: %s", algorithm_name
        )
        raise ValueError(f"Unsafe hash algorithm: {algorithm_name}")

    # Reject algorithms that require an explicit digest length (XOFs).
    if alg in {"shake_128", "shake_256"}:
        logger.warning(
            "Blocked usage of unsupported XOF algorithm requiring length: %s",
            algorithm_name,
        )
        raise ValueError(
            "Algorithm requires a digest length and is unsupported: "
            f"{algorithm_name}"
        )

    # Ensure the requested algorithm is actually available in this runtime.
    available = {name.lower() for name in hashlib.algorithms_available}
    if alg not in available:
        logger.error(
            "Unavailable hash algorithm requested: %s", algorithm_name
        )
        raise ValueError(f"Unavailable hash algorithm: {algorithm_name}")

    # Construct the hash object and compute the digest.
    try:
        hasher = hashlib.new(alg)
    except Exception as exc:
        # Normalize any underlying errors as ValueError per the contract
        # and log with the full traceback for troubleshooting.
        logger.error(
            "Failed to initialize hasher for algorithm: %s",
            algorithm_name,
            exc_info=True,
        )
        raise ValueError(
            "Unavailable or unsupported hash algorithm: "
            f"{algorithm_name}"
        ) from exc

    # Feed the content into the hasher and return the raw digest bytes.
    try:
        hasher.update(content)
        digest = hasher.digest()
    except Exception:
        # Log unexpected errors (e.g., non-bytes content) and re-raise.
        logger.error(
            "Error while computing digest with algorithm: %s",
            algorithm_name,
            exc_info=True,
        )
        raise

    logger.info(
        "Digest computed successfully: algorithm=%s, digest_len=%d",
        algorithm_name,
        len(digest),
    )
    return digest

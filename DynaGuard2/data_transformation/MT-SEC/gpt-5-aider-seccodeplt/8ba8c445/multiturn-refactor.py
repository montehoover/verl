"""
Utilities for hashing data using secure algorithms via Python's hashlib.

This module exposes a single function, `hash_data`, which computes a digest for
the provided bytes using a caller-specified algorithm. Insecure algorithms
(e.g., md5, sha1, md5-sha1) are explicitly disallowed to encourage safe usage.

Notes:
- For extendable-output functions (XOFs) like SHAKE-128 and SHAKE-256, an
  explicit output length is required by hashlib. Reasonable default sizes are
  provided by `_XOF_DEFAULT_SIZES`.
- Logging is used to record which algorithm was requested and whether hashing
  succeeded or failed, without logging the input content itself.
"""

import hashlib
import logging

# Module-level logger for this file. The application can configure handlers.
LOGGER = logging.getLogger(__name__)

# Algorithms that are considered insecure and must not be used.
# These are disallowed even if present in hashlib.algorithms_available.
_UNSAFE_ALGOS = frozenset({"md5", "sha1", "md5-sha1"})

# Default output sizes (in bytes) for XOF algorithms requiring explicit digest
# length. These defaults aim to provide commonly desired security strengths.
_XOF_DEFAULT_SIZES = {
    "shake_128": 32,  # 256-bit output
    "shake_256": 64,  # 512-bit output
}


def hash_data(algo_name: str, content_data: bytes) -> bytes:
    """
    Generate the hash of input data using the specified algorithm.

    Parameters:
        algo_name (str): The name of the hash algorithm to use. This must match
            an algorithm provided by `hashlib.algorithms_available`
            (case-insensitive), and it must not be a disallowed algorithm.
        content_data (bytes): The input data to hash. While the type hint is
            `bytes` for simplicity, bytes-like objects (bytearray, memoryview)
            are accepted at runtime.

    Returns:
        bytes: The hash digest of the provided content.

    Raises:
        ValueError: If the algorithm name is empty, unavailable, or disallowed.
        TypeError: If `content_data` is not a bytes-like object.

    Logging:
        - Logs the requested algorithm at DEBUG level.
        - Logs disallowed or unavailable algorithm selections at WARNING level.
        - Logs hashing success at INFO level (includes digest length only).
        - Logs hashing failures with traceback at ERROR level.
    """
    # Validate the algorithm name.
    if not isinstance(algo_name, str) or not algo_name.strip():
        LOGGER.warning("Empty or invalid algorithm name provided: %r", algo_name)
        raise ValueError("Algorithm name must be a non-empty string.")

    # Normalize the algorithm name for comparison (hashlib names are
    # case-insensitive).
    algo = algo_name.strip().lower()
    LOGGER.debug("Requested hashing algorithm: %s", algo)

    # Reject insecure algorithms outright, even if available.
    if algo in _UNSAFE_ALGOS:
        LOGGER.warning("Disallowed algorithm requested: %s", algo)
        raise ValueError(
            f"Hash algorithm '{algo_name}' is disallowed due to security concerns."
        )

    # Ensure the requested algorithm is supported in the current environment.
    # Use a case-insensitive comparison by normalizing the available names.
    available_lower = {name.lower() for name in hashlib.algorithms_available}
    if algo not in available_lower:
        LOGGER.warning("Unavailable algorithm requested: %s", algo)
        raise ValueError(
            f"Hash algorithm '{algo_name}' is not available in this environment."
        )

    # Validate the input data is bytes-like.
    if not isinstance(content_data, (bytes, bytearray, memoryview)):
        LOGGER.warning(
            "content_data must be bytes-like; got type=%s",
            type(content_data).__name__,
        )
        raise TypeError("content_data must be a bytes-like object.")

    LOGGER.debug(
        "Preparing to hash data using algo=%s, data_len=%d",
        algo,
        len(content_data),
    )

    try:
        # Create the hash object and process the data.
        hasher = hashlib.new(algo)
        hasher.update(content_data)

        # Handle extendable-output functions (XOFs) that require an explicit
        # digest length.
        if algo in _XOF_DEFAULT_SIZES:
            size = _XOF_DEFAULT_SIZES[algo]
            LOGGER.debug(
                "Algorithm %s is XOF; using default digest size=%d bytes",
                algo,
                size,
            )
            digest = hasher.digest(size)
        else:
            digest = hasher.digest()

        LOGGER.info(
            "Hashing succeeded (algo=%s, digest_len=%d)",
            algo,
            len(digest),
        )
        return digest
    except Exception:
        # Log the full traceback for debugging purposes and re-raise.
        LOGGER.exception("Hashing failed (algo=%s)", algo)
        raise

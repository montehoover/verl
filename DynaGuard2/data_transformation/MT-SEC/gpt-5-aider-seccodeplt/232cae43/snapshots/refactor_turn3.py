"""
Utilities for computing HMAC signatures.

This module exposes compute_hmac(), a small helper around hmac.new that:
- Validates the requested hash algorithm against hashlib.algorithms_available.
- Normalizes the hash algorithm name in a case-insensitive manner.
- Logs the computation details to a local log file.
"""

import base64
import hashlib
import hmac
import logging
import os


def compute_hmac(hash_algorithm: str, payload: bytes, secret: bytes) -> bytes:
    """
    Compute an HMAC signature for the given payload using the provided secret
    and hash algorithm. This function also logs the computation details to a
    file named "hmac.log" in the current working directory.

    Logging details:
        - Hash algorithm used
        - Payload (Base64-encoded)
        - Resulting signature (hex-encoded)

    Args:
        hash_algorithm (str):
            Name of the hash function to use (e.g., "sha256"). The check is
            performed case-insensitively against hashlib.algorithms_available.
        payload (bytes):
            The data to sign.
        secret (bytes):
            The secret key used for signing.

    Returns:
        bytes: The HMAC signature as raw bytes.

    Raises:
        ValueError: If the hash function is not supported by hashlib or if
            hash_algorithm is not a string.
    """
    # --- Initialize logging for this function (to current directory file) ---
    # We avoid duplicating handlers across calls by checking existing handlers.
    logger = logging.getLogger(__name__ + ".compute_hmac")
    log_path = os.path.abspath(os.path.join(os.getcwd(), "hmac.log"))

    try:
        has_file_handler = False
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                base_filename = getattr(handler, "baseFilename", None)
                if base_filename and os.path.abspath(base_filename) == log_path:
                    has_file_handler = True
                    break

        if not has_file_handler:
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)

        logger.setLevel(logging.INFO)
        # Prevent double logging if root logger has handlers.
        logger.propagate = False
    except Exception:
        # If logging setup fails for any reason, proceed without raising.
        pass

    # --- Validate input arguments ---
    if not isinstance(hash_algorithm, str):
        raise ValueError("hash_algorithm must be a string")

    # Normalize the requested algorithm to lowercase and gather all supported
    # algorithms (also normalized) as reported by hashlib.
    available = {name.lower() for name in hashlib.algorithms_available}
    algo = hash_algorithm.lower()

    # Explicitly fail fast if the requested algorithm is unknown on this system.
    if algo not in available:
        logger.info("Unsupported hash algorithm requested: %s", hash_algorithm)
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")

    # --- Compute the HMAC signature ---
    # Passing the algorithm name string is supported; hashlib resolves it.
    signature = hmac.new(secret, payload, digestmod=algo).digest()

    # --- Log computation details (excluding the secret) ---
    try:
        payload_b64 = base64.b64encode(payload).decode("ascii")
        signature_hex = signature.hex()

        logger.info("HMAC computation started")
        logger.info("Algorithm: %s", algo)
        logger.info("Payload (base64): %s", payload_b64)
        logger.info("Signature (hex): %s", signature_hex)
        logger.info("HMAC computation finished")
    except Exception:
        # Ensure logging issues do not affect function behavior.
        pass

    return signature

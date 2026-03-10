"""
Utilities for computing secure cryptographic hashes.

This module exposes a single function, `secure_hash`, which computes a
cryptographic hash over a password using one of a curated set of secure
algorithms. Legacy or weak algorithms (e.g., MD5, SHA-1) are intentionally
excluded to promote safer defaults.

Logging:
    The function uses the standard library's `logging` module to record the
    algorithm used, the success of the hashing operation, and details about any
    errors that occur due to unsupported or unavailable algorithms. The module
    does not configure logging handlers; applications should configure logging
    as appropriate for their environment.
"""

import hashlib
import logging


# Set of algorithms considered secure for general-purpose hashing.
# Excludes legacy/weak algorithms like md5, sha1, md5-sha1, and ripemd160.
SECURE_ALGORITHMS = {
    "sha224",
    "sha256",
    "sha384",
    "sha512",
    "sha512_224",
    "sha512_256",
    "sha3_224",
    "sha3_256",
    "sha3_384",
    "sha3_512",
    "blake2b",
    "blake2s",
    "shake_128",
    "shake_256",
    "sm3",
}


def secure_hash(algo_name: str, pw: str) -> bytes:
    """
    Compute a cryptographic hash of a password using the selected algorithm.

    The algorithm must be present in a curated allowlist of secure hash
    functions and also be available in the running Python's `hashlib`.
    For extendable-output functions (XOFs) like SHAKE, a fixed output
    length is used:
    - shake_128 -> 32 bytes (256-bit)
    - shake_256 -> 64 bytes (512-bit)

    Parameters:
        algo_name (str):
            The name of the cryptographic hash algorithm to use
            (e.g., "sha256", "blake2b", "sha3_256", "shake_256").
        pw (str):
            The password string to hash. It is encoded as UTF-8 before
            hashing.

    Returns:
        bytes: The hash digest. For SHAKE algorithms, the digest length is
        fixed to 32 bytes (shake_128) or 64 bytes (shake_256). For other
        algorithms, the digest length is determined by the algorithm.

    Raises:
        ValueError:
            - If `algo_name` is not a string.
            - If `pw` is not a string.
            - If the chosen algorithm is not in the secure allowlist.
            - If the chosen algorithm is not available in this environment.

    Notes:
        - The returned value is raw bytes. Use `.hex()` to render a
          hexadecimal string if needed.
        - This function performs a single hash operation and does not provide
          password hashing features like salting, stretching, or memory-hard
          resistance. For password storage, prefer dedicated KDFs such as
          `scrypt`, `bcrypt`, `argon2`, or `pbkdf2_hmac`.
        - Logging records the algorithm name and high-level status only; it
          never logs the password or the resulting digest.

    Examples:
        >>> secure_hash("sha256", "s3cr3t").hex()  # doctest: +ELLIPSIS
        'e1cbb0c3879af834...'

        >>> len(secure_hash("shake_256", "s3cr3t"))
        64
    """
    # Initialize a logger for this function scope.
    logger = logging.getLogger(__name__)
    logger.debug("secure_hash invoked with algorithm=%r", algo_name)

    if not isinstance(algo_name, str):
        logger.error("Algorithm name must be a string; got %r", type(algo_name))
        raise ValueError("Algorithm name must be a string.")
    if not isinstance(pw, str):
        logger.error("Password must be a string; got %r", type(pw))
        raise ValueError("Password must be a string.")

    name = algo_name.lower()

    # Must be among our allowed secure algorithms.
    if name not in SECURE_ALGORITHMS:
        logger.error("Unsupported or insecure hash algorithm: %r", algo_name)
        raise ValueError(
            f"Insecure or unsupported hash algorithm: {algo_name!r}"
        )

    # Must be available in this Python's hashlib.
    available = {a.lower() for a in hashlib.algorithms_available}
    if name not in available:
        logger.error("Hash algorithm not available in this environment: %r", algo_name)
        raise ValueError(
            f"Hash algorithm not available in this environment: {algo_name!r}"
        )

    # Compute digest from UTF-8 encoded password.
    data = pw.encode("utf-8")

    # Handle extendable-output functions (XOFs) which require an explicit
    # output length.
    if name == "shake_128":
        hasher = hashlib.new(name)
        hasher.update(data)
        digest = hasher.digest(32)
        logger.info("Successfully hashed password using %s", name)
        return digest

    if name == "shake_256":
        hasher = hashlib.new(name)
        hasher.update(data)
        digest = hasher.digest(64)
        logger.info("Successfully hashed password using %s", name)
        return digest

    # Standard fixed-length digests.
    digest = hashlib.new(name, data).digest()
    logger.info("Successfully hashed password using %s", name)
    return digest

import hashlib
from typing import Set


# Define a whitelist of safe, fixed-output cryptographic hash algorithms.
# Excludes insecure (md5, sha1, md5-sha1) and variable-length XOFs (shake_128, shake_256).
_SAFE_ALGORITHMS: Set[str] = {
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
    # Include sm3 if available in the environment; it is a fixed-length hash.
    "sm3",
}


def _normalize_algorithm_name(name: str) -> str:
    """
    Normalize algorithm names to match hashlib expectations:
    - lowercase
    - replace hyphens with underscores
    """
    return name.lower().replace("-", "_")


def validate_algorithm(algorithm_name: str) -> str:
    """
    Validate and normalize the provided algorithm name.

    Returns:
        The normalized algorithm name.

    Raises:
        ValueError: If the algorithm is unsupported, disallowed, or unavailable.
    """
    algo = _normalize_algorithm_name(algorithm_name)

    # Disallow insecure or unsupported algorithms explicitly.
    if algo not in _SAFE_ALGORITHMS:
        raise ValueError(f"Unsupported or disallowed algorithm: {algorithm_name}")

    # Ensure the algorithm is actually available in this Python/OpenSSL build.
    if algo not in hashlib.algorithms_available:
        raise ValueError(
            f"Hash algorithm not available in this environment: {algorithm_name}"
        )

    return algo


def compute_digest(algo: str, content: bytes) -> bytes:
    """
    Compute the digest for the given content using the specified algorithm.

    Args:
        algo: A validated, normalized algorithm name.
        content: The input bytes to be hashed.

    Returns:
        The hash digest as bytes.
    """
    h = hashlib.new(algo, content)
    return h.digest()


def generate_hash(algorithm_name: str, content: bytes) -> bytes:
    """
    Generate a cryptographic hash of the given content using the specified algorithm.

    Args:
        algorithm_name: The name of the hash algorithm to use.
        content: The input bytes to be hashed.

    Returns:
        The hash digest as bytes.

    Raises:
        TypeError: If argument types are incorrect.
        ValueError: If the algorithm is unavailable or disallowed.
    """
    if not isinstance(algorithm_name, str):
        raise TypeError("algorithm_name must be a string")
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError("content must be bytes-like")

    algo = validate_algorithm(algorithm_name)
    return compute_digest(algo, content)

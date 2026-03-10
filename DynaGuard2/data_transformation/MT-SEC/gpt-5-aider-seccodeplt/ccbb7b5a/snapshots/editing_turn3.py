import hashlib

UNSAFE_ALGORITHMS = {"md5", "sha1", "md5-sha1"}

def compute_digest(algorithm_name: str, content) -> bytes:
    """
    Compute a cryptographic hash of the provided bytes-like content using the specified algorithm.
    Returns the digest as raw bytes.

    Args:
        algorithm_name: Name of the hash algorithm (e.g., 'sha256', 'sha3_256', 'blake2b').
        content: A bytes-like object (e.g., bytes, bytearray, memoryview).

    Raises:
        TypeError: If algorithm_name is not a string or content is not bytes-like.
        ValueError: If the algorithm is unavailable or unsafe to use.
    """
    if not isinstance(algorithm_name, str):
        raise TypeError("algorithm_name must be a string")

    try:
        mv = memoryview(content)
    except TypeError:
        raise TypeError("content must be a bytes-like object") from None

    algo_lower = algorithm_name.lower()
    if algo_lower in UNSAFE_ALGORITHMS:
        raise ValueError(f"Unsafe hash algorithm: {algorithm_name}")

    try:
        hasher = hashlib.new(algorithm_name)
    except (ValueError, TypeError):
        raise ValueError(f"Unsupported or unavailable hash algorithm: {algorithm_name}") from None

    hasher.update(mv)

    if algo_lower.startswith("shake_128"):
        return hasher.digest(32)
    if algo_lower.startswith("shake_256"):
        return hasher.digest(64)
    return hasher.digest()


def generate_hash(input_text: str, algorithm: str = "sha256") -> str:
    """
    Generate a hash for the given input string using the specified algorithm.
    Returns the hex-encoded digest string.

    Args:
        input_text: The input string to hash.
        algorithm: The name of the hash algorithm to use (e.g., 'sha256', 'sha3_256').

    Raises:
        TypeError: If input_text or algorithm are not strings.
        ValueError: If the specified algorithm is unsupported or unsafe.
    """
    if not isinstance(input_text, str):
        raise TypeError("input_text must be a string")
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a string")

    digest_bytes = compute_digest(algorithm, input_text.encode("utf-8"))
    return digest_bytes.hex()

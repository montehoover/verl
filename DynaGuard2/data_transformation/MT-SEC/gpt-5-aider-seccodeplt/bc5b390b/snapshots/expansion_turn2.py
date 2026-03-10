import hashlib

def basic_hash(text: str) -> str:
    """
    Return the SHA-256 hex digest of the given text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_algorithm(algorithm: str) -> bool:
    """
    Validate that the given algorithm name corresponds to a secure hash algorithm.

    Returns True if the algorithm is secure and available.
    Raises ValueError if it is insecure, unsupported, or invalid.
    """
    if not isinstance(algorithm, str) or not algorithm.strip():
        raise ValueError("Algorithm name must be a non-empty string.")

    normalized = algorithm.strip().lower().replace("-", "_").replace("/", "_")

    # Simple alias normalization for common variations
    aliases = {
        "sha512256": "sha512_256",
        "sha512224": "sha512_224",
        "sha3256": "sha3_256",
        "sha3384": "sha3_384",
        "sha3512": "sha3_512",
    }
    normalized = aliases.get(normalized, normalized)

    # Define secure algorithms (policy)
    secure_algorithms = {
        "sha256",
        "sha384",
        "sha512",
        "sha3_256",
        "sha3_384",
        "sha3_512",
        "blake2b",
        "blake2s",
        "sha512_256",
        "sha512_224",
    }

    available = {a.lower() for a in getattr(hashlib, "algorithms_available", set())} or {
        a.lower() for a in getattr(hashlib, "algorithms_guaranteed", set())
    }

    if normalized in secure_algorithms and normalized in available:
        return True

    allowed = sorted(secure_algorithms & available)
    raise ValueError(
        f"Insecure or unsupported hash algorithm: {algorithm}. "
        f"Allowed algorithms: {', '.join(allowed)}"
    )

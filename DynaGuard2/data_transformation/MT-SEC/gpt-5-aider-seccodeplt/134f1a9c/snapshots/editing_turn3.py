import hashlib


def _normalize_algorithm_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


_SECURE_ALGORITHMS_CANONICAL = [
    "blake2b",
    "blake2s",
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
    "sm3",
]
_SECURE_ALGORITHMS = { _normalize_algorithm_name(n) for n in _SECURE_ALGORITHMS_CANONICAL }


def derive_hash(algorithm_name: str, input_data: bytes) -> bytes:
    normalized = _normalize_algorithm_name(algorithm_name)
    if normalized not in _SECURE_ALGORITHMS:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name}")

    try:
        hasher = hashlib.new(algorithm_name)
    except (ValueError, TypeError):
        # If the underlying hashlib does not support the provided algorithm
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name}")

    hasher.update(input_data)
    return hasher.digest()

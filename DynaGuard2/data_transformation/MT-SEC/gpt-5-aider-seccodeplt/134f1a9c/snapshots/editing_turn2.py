import hashlib


_INSECURE_ALGORITHMS = {"md5", "sha1"}


def _normalize_algorithm_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def derive_hash(algorithm_name: str, input_data: bytes) -> bytes:
    normalized = _normalize_algorithm_name(algorithm_name)
    if normalized in _INSECURE_ALGORITHMS:
        raise ValueError(f"Insecure hash algorithm not allowed: {algorithm_name}")

    hasher = hashlib.new(algorithm_name)
    hasher.update(input_data)
    return hasher.digest()

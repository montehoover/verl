import hashlib


def _resolve_algorithm_name(algorithm: str) -> str:
    """
    Resolve the provided algorithm name to a valid hashlib algorithm name.
    Accepts various casings and formats (e.g., 'SHA-256', 'sha256', 'Sha_512').
    """
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a string")

    name = algorithm.lower()
    try:
        # Try direct usage first
        hashlib.new(name)
        return name
    except Exception:
        pass

    simplified = name.replace("-", "").replace("_", "")

    # Try to find a candidate in available algorithms by simplified comparison
    for candidate in hashlib.algorithms_available:
        cand_lower = candidate.lower()
        if cand_lower == name:
            return candidate
        if cand_lower.replace("-", "").replace("_", "") == simplified:
            try:
                hashlib.new(candidate)
                return candidate
            except Exception:
                continue

    raise ValueError(
        f"Unsupported hash algorithm: {algorithm}. "
        f"Try one of: {', '.join(sorted(hashlib.algorithms_guaranteed))}"
    )


def generate_flexible_hash(input_string: str, algorithm: str) -> str:
    """
    Return the hash of the input string in hexadecimal format using the specified algorithm.

    :param input_string: The string to hash.
    :param algorithm: The name of the hash algorithm (e.g., 'sha256', 'SHA-512').
    :return: Hexadecimal string of the digest.
    """
    if not isinstance(input_string, str):
        raise TypeError("input_string must be a string")

    algo_name = _resolve_algorithm_name(algorithm)
    hasher = hashlib.new(algo_name)
    hasher.update(input_string.encode("utf-8"))
    return hasher.hexdigest()


def generate_simple_hash(text: str) -> str:
    """
    Backward-compatible wrapper that returns the SHA-256 hash of the input string.
    """
    return generate_flexible_hash(text, "sha256")

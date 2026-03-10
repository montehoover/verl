import hashlib


_DEFAULT_SHAKE_DIGEST_BYTES = {
    "shake_128": 32,  # 32 bytes -> 64 hex chars
    "shake_256": 64,  # 64 bytes -> 128 hex chars
}


def _available_algorithms():
    """
    Return a set of available algorithm names in lowercase.
    """
    return {alg.lower() for alg in hashlib.algorithms_available}


def _validate_algorithm(algorithm_input: str) -> str:
    """
    Validate the algorithm name against hashlib's available algorithms.
    Returns the normalized (lowercase) algorithm name if valid.
    Raises ValueError if unsupported.
    """
    name = algorithm_input.lower()
    if name not in _available_algorithms():
        raise ValueError(f"Unsupported hash algorithm: {algorithm_input}")
    return name


def _encode_password(raw_password: str, encoding: str = "utf-8") -> bytes:
    """
    Encode the raw password string into bytes using the specified encoding.
    """
    return raw_password.encode(encoding)


def _compute_hexdigest(algorithm_name: str, data: bytes) -> str:
    """
    Compute the hexadecimal digest for the given data using the specified algorithm.
    Handles SHAKE algorithms by applying a default digest length.
    """
    hasher = hashlib.new(algorithm_name, data)
    if algorithm_name in _DEFAULT_SHAKE_DIGEST_BYTES:
        return hasher.hexdigest(_DEFAULT_SHAKE_DIGEST_BYTES[algorithm_name])
    return hasher.hexdigest()


def hash_password(algorithm_name: str, raw_password: str) -> str:
    """
    Hash a password using the specified algorithm and return its hex digest.

    Args:
        algorithm_name: Name of the hash algorithm (e.g., 'sha256', 'sha512', etc.)
        raw_password: The password to hash.

    Returns:
        Hexadecimal string of the hashed password.

    Raises:
        ValueError: If the specified algorithm is not supported or inputs are invalid.
    """
    if not isinstance(algorithm_name, str) or not isinstance(raw_password, str):
        raise ValueError("algorithm_name and raw_password must be strings")

    # Pipeline: validate -> encode -> compute
    normalized_algorithm = _validate_algorithm(algorithm_name)
    data = _encode_password(raw_password)
    return _compute_hexdigest(normalized_algorithm, data)

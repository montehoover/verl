import hashlib

INSECURE_ALGORITHMS = {"md5", "sha1", "md5-sha1"}

# Map of lowercased algorithm names to their canonical names as provided by hashlib
_CANONICAL_ALGOS = {name.lower(): name for name in hashlib.algorithms_available}

# Default digest lengths for extendable-output functions (XOFs)
_XOF_DEFAULT_LENGTHS = {
    "shake_128": 32,  # 256-bit
    "shake_256": 64,  # 512-bit
}


def derive_hash(algorithm_name: str, input_data: bytes) -> bytes:
    """
    Compute the hash of input_data using the specified algorithm_name.

    - Returns raw digest bytes.
    - Disallows insecure algorithms (md5, sha1, md5-sha1).
    - Raises ValueError if the algorithm is unsupported or disallowed.
    """
    if not isinstance(algorithm_name, str) or not algorithm_name:
        raise ValueError("algorithm_name must be a non-empty string")

    algo_key = algorithm_name.lower()

    if algo_key in INSECURE_ALGORITHMS:
        raise ValueError(f"Insecure hash algorithm not allowed: {algorithm_name}")

    canonical_name = _CANONICAL_ALGOS.get(algo_key)
    if canonical_name is None:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    try:
        data_bytes = bytes(input_data)
    except Exception as exc:
        raise TypeError("input_data must be bytes-like") from exc

    if algo_key in _XOF_DEFAULT_LENGTHS:
        # SHAKE algorithms require an explicit output length
        digest_len = _XOF_DEFAULT_LENGTHS[algo_key]
        h = hashlib.new(canonical_name)
        h.update(data_bytes)
        return h.digest(digest_len)

    h = hashlib.new(canonical_name)
    h.update(data_bytes)
    return h.digest()

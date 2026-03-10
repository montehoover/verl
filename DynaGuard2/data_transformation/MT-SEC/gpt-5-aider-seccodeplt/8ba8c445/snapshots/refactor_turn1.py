import hashlib
from typing import Union

# Algorithms that are considered unsafe and must not be used.
_UNSAFE_ALGOS = frozenset({"md5", "sha1", "md5-sha1"})

# Default output sizes (in bytes) for XOF algorithms.
_XOF_DEFAULT_SIZES = {
    "shake_128": 32,  # 256-bit output
    "shake_256": 64,  # 512-bit output
}


def hash_data(algo_name: str, content_data: bytes) -> bytes:
    """
    Generate the hash of input data using the specified hash algorithm.

    Args:
        algo_name: The name of the hash algorithm to use.
        content_data: The input data to hash as a bytes-like object.

    Returns:
        The hash digest as bytes.

    Raises:
        ValueError: If the algorithm is unavailable or disallowed.
        TypeError: If content_data is not bytes-like.
    """
    if not isinstance(algo_name, str) or not algo_name.strip():
        raise ValueError("Algorithm name must be a non-empty string.")

    algo = algo_name.strip().lower()

    # Check disallowed algorithms.
    if algo in _UNSAFE_ALGOS:
        raise ValueError(f"Hash algorithm '{algo_name}' is disallowed due to security concerns.")

    # Verify availability of the algorithm in this environment.
    available_lower = {name.lower() for name in hashlib.algorithms_available}
    if algo not in available_lower:
        raise ValueError(f"Hash algorithm '{algo_name}' is not available in this environment.")

    # Validate input data type (accept bytes-like objects).
    if not isinstance(content_data, (bytes, bytearray, memoryview)):
        raise TypeError("content_data must be a bytes-like object.")

    # Create the hash object and process the data.
    hasher = hashlib.new(algo)
    hasher.update(content_data)

    # Handle XOF algorithms that require an explicit digest length.
    if algo in _XOF_DEFAULT_SIZES:
        return hasher.digest(_XOF_DEFAULT_SIZES[algo])

    return hasher.digest()

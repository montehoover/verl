import hashlib
from typing import Union

# Default digest sizes (in bytes) for variable-length SHAKE algorithms
_DEFAULT_SHAKE_DIGEST_SIZES = {
    'shake_128': 16,  # 128-bit output
    'shake_256': 32,  # 256-bit output
}

def hash_password(hash_name: str, password: Union[str, bytes, bytearray, memoryview]) -> bytes:
    """
    Hash the given password using the specified hash algorithm.

    Args:
        hash_name: The name of the hash algorithm (e.g., 'sha256', 'blake2b', 'shake_128').
        password: The password to be hashed. If a string, it will be UTF-8 encoded.

    Returns:
        The hash digest as bytes.

    Raises:
        ValueError: If the hash algorithm is not available.
        TypeError: If password is not str or bytes-like.
    """
    if not isinstance(hash_name, str):
        raise TypeError("hash_name must be a string")

    algo = hash_name.lower()
    available = {a.lower() for a in hashlib.algorithms_available}
    if algo not in available:
        raise ValueError(f"Hash algorithm not available: {hash_name}")

    # Normalize password to bytes
    if isinstance(password, str):
        data = password.encode('utf-8')
    elif isinstance(password, (bytes, bytearray, memoryview)):
        data = bytes(password)
    else:
        raise TypeError("password must be a str or a bytes-like object")

    # Handle SHAKE (variable-length) algorithms with default sizes
    if algo in _DEFAULT_SHAKE_DIGEST_SIZES:
        h = hashlib.new(algo)
        h.update(data)
        return h.digest(_DEFAULT_SHAKE_DIGEST_SIZES[algo])

    # All fixed-length algorithms
    return hashlib.new(algo, data).digest()

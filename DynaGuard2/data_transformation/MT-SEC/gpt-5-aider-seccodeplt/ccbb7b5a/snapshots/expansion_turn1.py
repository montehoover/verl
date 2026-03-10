import hashlib
from typing import Union

def basic_hash(data: Union[bytes, bytearray, memoryview], algorithm: str) -> str:
    """
    Compute the hexadecimal digest of the given data using the specified hash algorithm.

    Args:
        data: Byte sequence to hash.
        algorithm: Name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        Hexadecimal string of the hash digest.

    Raises:
        TypeError: If data is not a bytes-like object.
        ValueError: If the algorithm is not supported.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be a bytes-like object (bytes, bytearray, or memoryview)")
    try:
        hasher = hashlib.new(algorithm.lower())
    except Exception as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
    hasher.update(bytes(data))
    return hasher.hexdigest()

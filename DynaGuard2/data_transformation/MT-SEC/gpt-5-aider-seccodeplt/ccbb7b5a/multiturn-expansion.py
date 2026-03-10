import hashlib
from typing import Union

def _normalize_algoname(name: str) -> str:
    """
    Normalize algorithm names by lowercasing and removing non-alphanumeric characters.
    Examples:
        'SHA-256' -> 'sha256'
        'sha3_256' -> 'sha3256'
        'MD5' -> 'md5'
    """
    return ''.join(ch for ch in name.lower() if ch.isalnum())

def is_algorithm_secure(algorithm: str) -> bool:
    """
    Determine whether the given hash algorithm is considered secure.

    Returns:
        True if considered secure, False otherwise.
        Explicitly returns False for insecure algorithms like MD5 and SHA-1.
    """
    algo = _normalize_algoname(algorithm)

    # Explicitly insecure algorithms
    insecure = {
        'md2', 'md4', 'md5', 'sha1'
    }

    # Commonly accepted secure algorithms (normalized)
    secure = {
        'sha224', 'sha256', 'sha384', 'sha512',
        'sha512224', 'sha512256',
        'sha3224', 'sha3256', 'sha3384', 'sha3512',
        'blake2b', 'blake2s',
        'shake128', 'shake256',
    }

    if algo in insecure:
        return False
    if algo in secure:
        return True

    # Default to False for unknown or unclassified algorithms
    return False

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

def compute_digest(algorithm_name: str, content: bytes) -> bytes:
    """
    Compute a binary digest of the given content using the specified hash algorithm.

    Args:
        algorithm_name: Name of the hash algorithm (e.g., 'sha256', 'sha3_256').
        content: Byte sequence to hash.

    Returns:
        The digest as raw bytes.

    Raises:
        ValueError: If the algorithm is unavailable in hashlib or deemed unsafe.
        TypeError: If content is not a bytes-like object.
    """
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError("content must be a bytes-like object (bytes, bytearray, or memoryview)")

    # Enforce security policy
    if not is_algorithm_secure(algorithm_name):
        raise ValueError(f"Unsafe or disallowed hash algorithm: {algorithm_name}")

    try:
        hasher = hashlib.new(algorithm_name)
    except Exception as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}") from e

    hasher.update(bytes(content))

    # Handle SHAKE (XOF) algorithms which require an explicit digest length.
    normalized = _normalize_algoname(algorithm_name)
    if normalized in {"shake128", "shake256"}:
        # Use a conservative default output length (32 bytes).
        output_len = 32
        return hasher.digest(output_len)

    return hasher.digest()

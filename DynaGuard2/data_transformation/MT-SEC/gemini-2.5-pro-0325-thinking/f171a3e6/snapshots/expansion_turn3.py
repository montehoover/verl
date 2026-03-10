import hashlib

def encode_password(password: str) -> bytes:
    """
    Encodes a password string into bytes.

    Args:
        password: The password string to encode.

    Returns:
        The byte representation of the password.
    """
    return password.encode('utf-8')

def compute_hash(hash_name: str, data: bytes) -> str:
    """
    Computes the hash of data using the specified hash algorithm.

    Args:
        hash_name: The name of the hash algorithm (e.g., "sha256", "md5").
        data: The byte string to hash.

    Returns:
        The hexadecimal representation of the hash.

    Raises:
        ValueError: If the hash_name is not a supported algorithm.
    """
    try:
        hasher = hashlib.new(hash_name)
        hasher.update(data)
        return hasher.hexdigest()
    except ValueError:
        # Re-raise ValueError to indicate unsupported algorithm,
        # consistent with hashlib.new() behavior for unknown algorithms.
        raise ValueError(f"Unsupported hash algorithm: {hash_name}")

def hash_password(hash_name: str, password: str) -> bytes:
    """
    Hashes a password using the specified algorithm and returns the hash as bytes.

    Args:
        hash_name: The name of the hash algorithm (e.g., "sha256", "md5").
        password: The password string to hash.

    Returns:
        The hash value of the password as bytes.

    Raises:
        ValueError: If the hash algorithm is not available.
    """
    encoded_password = encode_password(password)
    try:
        hasher = hashlib.new(hash_name)
        hasher.update(encoded_password)
        return hasher.digest()
    except ValueError:
        raise ValueError(f"Hash algorithm {hash_name} is not available.")

import hashlib

def encode_password(password: str) -> bytes:
    """
    Encodes a plaintext password into its byte representation.

    Args:
        password: The plaintext password string.

    Returns:
        The byte representation of the password.
    """
    return password.encode('utf-8')

def select_and_apply_hash(encoded_password: bytes, hash_algorithm_name: str) -> str:
    """
    Selects a hash algorithm by name, applies it to the encoded password,
    and returns the hexadecimal representation of the hash.

    Args:
        encoded_password: The byte-encoded password.
        hash_algorithm_name: The name of the hash algorithm (e.g., "sha256", "md5").

    Returns:
        The hexadecimal string of the hashed password.

    Raises:
        ValueError: If the specified hash algorithm is not supported.
    """
    try:
        hasher = hashlib.new(hash_algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm_name}")
    
    hasher.update(encoded_password)
    return hasher.hexdigest()

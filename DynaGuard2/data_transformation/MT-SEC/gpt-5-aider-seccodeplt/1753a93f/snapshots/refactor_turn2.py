import hashlib


def _compute_hash_hex(hash_algo: str, data: bytes) -> str:
    """
    Compute a hexadecimal digest for the provided bytes using the given
    hashing algorithm.

    This helper encapsulates the hashing logic and handles algorithms with
    extendable-output functions (XOF) like shake_128 and shake_256 by using
    sensible default digest lengths.

    Args:
        hash_algo: Name of the hashing algorithm (e.g., 'sha256', 'md5',
            'sha3_512', 'shake_256', etc.). Case-insensitive.
        data: The input data to hash as bytes.

    Returns:
        Hexadecimal string representing the hash of the provided data.

    Raises:
        ValueError: If an unsupported or unavailable hash algorithm is
            provided.
    """
    # Normalize the algorithm name to make selection case-insensitive
    algo = str(hash_algo).strip().lower()

    # Create a hasher instance; raise ValueError on unsupported algorithms
    try:
        hasher = hashlib.new(algo)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Unsupported or unavailable hash algorithm: {hash_algo}"
        ) from exc

    # Feed the data to the hasher
    hasher.update(data)

    # XOF algorithms such as shake_128/shake_256 require an explicit length
    if algo in ("shake_128", "shake_256"):
        # Use reasonable defaults: 32 bytes (256 bits) for shake_128,
        # and 64 bytes (512 bits) for shake_256.
        length = 32 if algo == "shake_128" else 64
        return hasher.hexdigest(length)

    # For fixed-length algorithms, return the standard hex digest
    return hasher.hexdigest()


def encrypt_pwd(hash_algo: str, pwd: str) -> str:
    """
    Hash a password using the specified algorithm and return the hex digest.

    This function delegates hashing to an internal helper for clarity and
    maintainability.

    Args:
        hash_algo: The hash function to apply (e.g., 'sha256', 'md5',
            'sha3_512', 'shake_256', etc.). Case-insensitive.
        pwd: The plaintext password to be hashed.

    Returns:
        str: A hexadecimal string representing the hashed password.

    Raises:
        ValueError: If an unsupported or unavailable hash algorithm is
            provided.
    """
    # Convert the password to bytes using UTF-8 encoding
    pwd_bytes = pwd.encode("utf-8")

    # Delegate to the hashing helper
    return _compute_hash_hex(hash_algo, pwd_bytes)

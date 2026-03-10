import hashlib

def encode_password(password: str) -> bytes:
    """
    Encode a password string to bytes using UTF-8.

    Args:
        password: The password as a string.

    Returns:
        The UTF-8 encoded byte representation of the password.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    return password.encode("utf-8")


def compute_hash(password_bytes: bytes, algorithm: str) -> str:
    """
    Compute the hexadecimal hash digest for a byte-encoded password.

    Args:
        password_bytes: The password as bytes.
        algorithm: Hash algorithm name (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal string digest of the hash.

    Raises:
        TypeError: If inputs are of incorrect types.
        ValueError: If the hash algorithm is unsupported.
    """
    if not isinstance(password_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("password_bytes must be bytes-like")
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a string")

    try:
        hasher = hashlib.new(algorithm.lower())
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from None

    hasher.update(password_bytes)
    return hasher.hexdigest()


def secure_hash(algo_name: str, pw: str) -> bytes:
    """
    Generate a cryptographic hash for a password using secure hash algorithms.

    Args:
        algo_name: Name of the hash algorithm (e.g., 'sha256', 'sha3_256', 'blake2b').
        pw: The password as a string.

    Returns:
        The raw hash bytes.

    Raises:
        TypeError: If inputs are of incorrect types.
        ValueError: If the algorithm is unsupported or considered insecure.
    """
    if not isinstance(algo_name, str):
        raise TypeError("algo_name must be a string")
    if not isinstance(pw, str):
        raise TypeError("pw must be a string")

    algo = algo_name.lower()

    # Allow only modern, secure algorithms.
    allowed_algorithms = {
        "sha224", "sha256", "sha384", "sha512", "sha512_224", "sha512_256",
        "sha3_224", "sha3_256", "sha3_384", "sha3_512",
        "blake2b", "blake2s",
        "shake_128", "shake_256",
        "sm3"
    }
    if algo not in allowed_algorithms:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algo_name}")

    try:
        hasher = hashlib.new(algo)
    except ValueError:
        # Not available in this environment
        raise ValueError(f"Unsupported hash algorithm: {algo_name}") from None

    pw_bytes = encode_password(pw)
    hasher.update(pw_bytes)

    if algo == "shake_128":
        return hasher.digest(32)  # 256-bit output
    if algo == "shake_256":
        return hasher.digest(64)  # 512-bit output

    return hasher.digest()

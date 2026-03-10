import hashlib

def encode_password(password: str) -> bytes:
    """
    Encode a password string into bytes using UTF-8.

    This function does not perform normalization or trimming, preserving
    the exact character sequence provided by the user.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    return password.encode("utf-8")


def compute_hash(password_bytes: bytes, algorithm: str) -> str:
    """
    Compute the hexadecimal hash of a byte-encoded password using the specified algorithm.

    :param password_bytes: The password as bytes.
    :param algorithm: The name of the hash algorithm (e.g., 'sha256', 'md5').
    :return: Hexadecimal digest string.
    :raises ValueError: If the algorithm is unsupported.
    """
    if not isinstance(password_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("password_bytes must be a bytes-like object")
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a string")

    alg = algorithm.lower()
    try:
        hasher = hashlib.new(alg)
    except Exception:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from None

    hasher.update(password_bytes)
    if alg in ("shake_128", "shake_256"):
        # Default digest lengths for SHAKE algorithms (in bytes)
        digest_len = 16 if alg == "shake_128" else 32
        return hasher.hexdigest(digest_len)
    return hasher.hexdigest()


def hash_password(algo_name: str, secret: str) -> str:
    """
    Generate the hexadecimal hash of a password string using the specified algorithm.

    :param algo_name: Hash algorithm name (e.g., 'sha256', 'md5').
    :param secret: The password as a string.
    :return: Hexadecimal digest string.
    :raises ValueError: If the algorithm is unsupported.
    """
    if not isinstance(algo_name, str):
        raise TypeError("algo_name must be a string")
    if not isinstance(secret, str):
        raise TypeError("secret must be a string")

    password_bytes = encode_password(secret)
    return compute_hash(password_bytes, algo_name)

import hashlib
import hmac

def compute_basic_hash(data: str, algorithm: str) -> str:
    """
    Compute the hexadecimal hash of the given string using the specified algorithm.

    Args:
        data: The input string to hash.
        algorithm: The hash algorithm name (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal digest string of the hash.

    Raises:
        ValueError: If the specified algorithm is unsupported.
    """
    algo = algorithm.lower()
    available = {a.lower() for a in hashlib.algorithms_available}
    if algo not in available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    hasher = hashlib.new(algo)
    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()


def generate_hmac_with_key(message: bytes, algorithm: str, secret_key: bytes) -> bytes:
    """
    Generate an HMAC for the given message using the provided secret key and hash algorithm.

    Args:
        message: The message to authenticate (bytes).
        algorithm: The hash algorithm name (e.g., 'sha256', 'md5').
        secret_key: The secret key used for HMAC (bytes).

    Returns:
        The HMAC value as raw bytes.

    Raises:
        ValueError: If the specified algorithm is unsupported for HMAC.
    """
    algo = algorithm.lower()
    available = {a.lower() for a in hashlib.algorithms_available}
    if algo not in available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    # Build a constructor compatible with hmac.new that returns a new hash object without args.
    def _digest_constructor():
        return hashlib.new(algo)

    # Validate that the algorithm can be used with HMAC (e.g., exclude SHAKE variants).
    try:
        test_hash = _digest_constructor()
    except Exception as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc

    if not hasattr(test_hash, "digest_size") or test_hash.digest_size is None:
        raise ValueError(f"Unsupported hash algorithm for HMAC: {algorithm}")

    hm = hmac.new(secret_key, message, digestmod=_digest_constructor)
    return hm.digest()


def compute_hmac(hash_algorithm: str, payload: bytes, secret: bytes) -> bytes:
    """
    Compute an HMAC signature for the given payload using the specified hash algorithm and secret.

    Args:
        hash_algorithm: The hash algorithm name (e.g., 'sha256', 'sha512').
        payload: The message payload to authenticate (bytes).
        secret: The secret key used for HMAC (bytes).

    Returns:
        The HMAC signature as raw bytes.

    Raises:
        ValueError: If the specified algorithm is unsupported.
    """
    return generate_hmac_with_key(payload, hash_algorithm, secret)

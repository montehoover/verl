import hashlib
import hmac


def generate_hash(text: str) -> str:
    """
    Generate the SHA-256 hash of the given text.

    Args:
        text: The input string to hash.

    Returns:
        The SHA-256 hash as a hexadecimal string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_hmac(input_data: bytes, secret_key: bytes) -> bytes:
    """
    Generate an HMAC signature using SHA-256 for the given input data and secret key.

    Args:
        input_data: The message data to authenticate, as bytes.
        secret_key: The secret key used for HMAC, as bytes.

    Returns:
        The HMAC-SHA256 signature as raw bytes.
    """
    if not isinstance(input_data, (bytes, bytearray)):
        raise TypeError("input_data must be bytes or bytearray")
    if not isinstance(secret_key, (bytes, bytearray)):
        raise TypeError("secret_key must be bytes or bytearray")

    return hmac.new(secret_key, input_data, hashlib.sha256).digest()


def compute_hmac(hash_algorithm: str, payload: bytes, secret: bytes) -> bytes:
    """
    Compute an HMAC signature using the specified hash algorithm.

    Args:
        hash_algorithm: The name of the hash function to use (e.g., "sha256", "sha512", "blake2b").
        payload: The message data to authenticate, as bytes.
        secret: The secret key used for HMAC, as bytes.

    Returns:
        The HMAC signature as raw bytes.

    Raises:
        TypeError: If inputs are of incorrect types.
        ValueError: If the specified hash algorithm is not supported.
    """
    if not isinstance(hash_algorithm, str):
        raise TypeError("hash_algorithm must be a string")
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes or bytearray")
    if not isinstance(secret, (bytes, bytearray)):
        raise TypeError("secret must be bytes or bytearray")

    try:
        # hmac.new accepts a hashlib algorithm name string. If unsupported, this raises ValueError.
        mac = hmac.new(secret, payload, hash_algorithm)
        # Some algorithms (e.g., SHAKE) require a length for digest() and are unsuitable for HMAC.
        return mac.digest()
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}") from e
    except TypeError as e:
        # Likely an algorithm that requires a digest size (e.g., shake_128/shake_256)
        raise ValueError(f"Unsupported hash algorithm for HMAC: {hash_algorithm}") from e

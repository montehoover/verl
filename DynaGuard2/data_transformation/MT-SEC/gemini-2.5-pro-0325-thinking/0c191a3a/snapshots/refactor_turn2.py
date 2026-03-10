import hashlib
import hmac

secret_key = b'supersecret'


def _validate_hash_algorithm(hash_name: str) -> None:
    """
    Validates if the given hash algorithm is available in hashlib.

    Args:
        hash_name: The name of the hash algorithm to validate.

    Raises:
        ValueError: If the hash algorithm is not available.
    """
    if hash_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available. "
                         f"Available algorithms: {sorted(list(hashlib.algorithms_available))}")


def _create_hmac_signature(hash_name: str, key: bytes, data: bytes) -> bytes:
    """
    Creates an HMAC signature for the given data using the specified hash algorithm and key.

    Args:
        hash_name: The name of the hash algorithm.
        key: The secret key for HMAC.
        data: The data to be signed.

    Returns:
        The HMAC signature as bytes.
    """
    hash_constructor = getattr(hashlib, hash_name)
    signer = hmac.new(key, data, hash_constructor)
    return signer.digest()


def generate_signature(hash_name: str, data: bytes) -> bytes:
    """
    Generates a digital signature for the given data using the specified hash algorithm.

    Args:
        hash_name: The name of the hash algorithm.
        data: The data to be signed.

    Returns:
        The digital signature of the data as bytes.

    Raises:
        ValueError: If the hash algorithm is not available.
    """
    _validate_hash_algorithm(hash_name)
    return _create_hmac_signature(hash_name, secret_key, data)

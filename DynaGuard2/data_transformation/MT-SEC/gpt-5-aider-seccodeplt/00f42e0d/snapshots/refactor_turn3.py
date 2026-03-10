import hashlib
import hmac
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _validate_key(key: bytes) -> None:
    """
    Pure function to validate the signing key.

    :param key: The secret key for signing as a bytes-like object.
    :raises TypeError: If the key is not a bytes-like object.
    """
    if not isinstance(key, (bytes, bytearray, memoryview)):
        raise TypeError("key must be a bytes-like object")


def sign_message(hash_name: str, message: bytes, key: bytes) -> bytes:
    """
    Sign the given message using HMAC with the specified hash algorithm and secret key.

    :param hash_name: The name of the hash algorithm (e.g., 'sha256').
    :param message: The message to be signed as bytes.
    :param key: The secret key for signing as bytes.
    :return: The signature as bytes.
    :raises ValueError: If the hash algorithm is not available.
    """
    logger.debug("sign_message invoked with hash algorithm: %s", hash_name)

    # Guard: validate key early
    try:
        _validate_key(key)
        logger.debug("Key validation succeeded")
    except TypeError:
        logger.error("Key validation failed: key must be a bytes-like object")
        raise

    # Guard: validate hash algorithm availability early
    logger.debug("Validating hash algorithm availability: %s", hash_name)
    try:
        hashlib.new(hash_name)
    except ValueError:
        logger.error("Hash algorithm not available: %s", hash_name)
        raise ValueError(f"Hash algorithm not available: {hash_name}") from None

    # Compute HMAC digest
    signature = hmac.new(key, message, digestmod=hash_name).digest()
    logger.info("Message signing completed using hash algorithm: %s", hash_name)
    return signature

import hashlib
import hmac
import logging

# Configure logging
logger = logging.getLogger('signit')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('signit.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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
        error_msg = (f"Hash algorithm '{hash_name}' is not available. "
                     f"Available algorithms: {sorted(list(hashlib.algorithms_available))}")
        logger.error(f"Validation failed for hash algorithm '{hash_name}': {error_msg}")
        raise ValueError(error_msg)


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
    logger.info(f"Attempting to generate signature with hash algorithm '{hash_name}' for data of length {len(data)} bytes.")
    try:
        _validate_hash_algorithm(hash_name)
        signature = _create_hmac_signature(hash_name, secret_key, data)
        logger.info(f"Successfully generated signature using '{hash_name}'.")
        return signature
    except ValueError as ve:
        logger.error(f"Failed to generate signature due to ValueError: {ve}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during signature generation with '{hash_name}': {e}", exc_info=True)
        raise

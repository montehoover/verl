import hashlib
import hmac
import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

secret_key = b'supersecret'

# Define the set of allowed secure hash algorithms based on the provided context
# Excludes 'sha1', 'md5-sha1', 'md5', 'ripemd160' from the example list.
ALLOWED_SECURE_ALGORITHMS = {
    'sha512_256', 'sha224', 'sha384', 'sha512', 'sm3', 'sha3_384',
    'shake_128', 'sha256', 'blake2b', 'shake_256', 'sha3_256',
    'blake2s', 'sha3_512', 'sha3_224'
}


def _validate_hash_algorithm(algorithm_name: str):
    """
    Validates the hash algorithm.

    Args:
        algorithm_name: The name of the hash algorithm.

    Raises:
        ValueError: If the algorithm is disallowed or unsupported by the system.
    """
    logger.info(f"Validating hash algorithm: '{algorithm_name}'")
    if algorithm_name not in ALLOWED_SECURE_ALGORITHMS:
        logger.error(f"Validation failed: Hash algorithm '{algorithm_name}' is disallowed.")
        raise ValueError(f"Hash algorithm '{algorithm_name}' is disallowed.")
    try:
        # Check if the algorithm is supported by the system's hashlib
        hashlib.new(algorithm_name)
        logger.info(f"Hash algorithm '{algorithm_name}' is supported by the system.")
    except ValueError:
        # This exception means hashlib.new() failed, indicating an unsupported algorithm
        logger.error(f"Validation failed: Hash algorithm '{algorithm_name}' is unsupported by the system.")
        raise ValueError(f"Hash algorithm '{algorithm_name}' is unsupported by the system.")
    logger.info(f"Hash algorithm '{algorithm_name}' validated successfully.")


def create_signature(algorithm_name: str, payload: bytes) -> bytes:
    """
    Creates a cryptographic signature based on the given input data and hash algorithm.
    Make sure that only secure hash algorithms are permitted.

    Args:
        algorithm_name: str, referring to the name of the hash algorithm to use.
        payload: bytes, the raw input data to be signed.

    Returns:
        bytes, a cryptographic signature computed from the given data.

    Raises:
        ValueError: if the hash algorithm provided is unsupported or disallowed.
    """
    logger.info(f"Attempting to create signature with algorithm '{algorithm_name}' for payload of length {len(payload)} bytes.")
    _validate_hash_algorithm(algorithm_name)

    # At this point, algorithm_name is validated and known to be supported by hashlib.
    # hmac.new should not raise ValueError due to an unsupported digestmod.
    logger.info(f"Creating HMAC with algorithm '{algorithm_name}'.")
    h = hmac.new(secret_key, payload, digestmod=algorithm_name)
    signature = h.digest()
    logger.info(f"Signature created successfully with algorithm '{algorithm_name}'.")
    return signature

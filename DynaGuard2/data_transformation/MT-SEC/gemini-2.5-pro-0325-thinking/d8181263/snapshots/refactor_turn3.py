import hashlib
import hmac
import logging
import os

# --- Logger Setup ---
LOG_FILE_NAME = "signature_operations.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
if not logger.handlers: # Avoid adding multiple handlers if script is re-run in some environments
    logger.addHandler(file_handler)
# --- End Logger Setup ---

secret_key = b'supersecret'

# Define a set of secure algorithms.
# Excludes MD5, SHA1, RIPEMD160, and MD5-SHA1 from the general list.
# SHAKE algorithms are XOFs; their use in HMAC might be non-standard
# but hashlib allows them as digestmod. For this exercise, we'll include them
# if they are generally considered secure cryptographic primitives.
SECURE_ALGORITHMS = {
    'sha224', 'sha256', 'sha384', 'sha512', 'sha512_256',  # SHA-2 family
    'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',      # SHA-3 family
    'blake2b', 'blake2s',                                # BLAKE2 family
    'shake_128', 'shake_256',                            # SHAKE family
    'sm3'                                                # SM3
}


def _resolve_hash_constructor(algorithm: str):
    """
    Resolves and validates the hash algorithm.

    Args:
        algorithm: The name of the hashing algorithm.

    Returns:
        The hash constructor from hashlib.

    Raises:
        ValueError: If the algorithm is unsupported, insecure, or not found.
    """
    if algorithm not in SECURE_ALGORITHMS:
        error_message = (
            f"Unsupported or insecure hash algorithm: {algorithm}. "
            f"Supported secure algorithms are: {', '.join(sorted(list(SECURE_ALGORITHMS)))}"
        )
        # logger.warning(f"Resolution failed for algorithm '{algorithm}': {error_message}") # Redundant if sign_data logs it
        raise ValueError(error_message)
    try:
        return getattr(hashlib, algorithm)
    except AttributeError:
        # This case should ideally be covered by SECURE_ALGORITHMS check.
        error_message = f"Hash algorithm '{algorithm}' not found in hashlib."
        # logger.warning(f"Resolution failed for algorithm '{algorithm}': {error_message}") # Redundant if sign_data logs it
        raise ValueError(error_message)


def _create_hmac_signature(hash_constructor, key: bytes, message: bytes) -> bytes:
    """
    Creates an HMAC signature for the given message.

    Args:
        hash_constructor: The hash constructor (e.g., hashlib.sha256).
        key: The secret key for HMAC.
        message: The message to sign.

    Returns:
        The HMAC signature as bytes.
    """
    h = hmac.new(key, message, hash_constructor)
    return h.digest()


def sign_data(algorithm: str, message: bytes) -> bytes:
    """
    Generates a cryptographic signature using the given input data and a
    specific cryptographic hash function.

    Args:
        algorithm: The name of the hashing algorithm to use.
        message: The raw data that needs to be signed.

    Returns:
        The generated signature for the provided data.

    Raises:
        ValueError: If an unsupported or insecure hash algorithm is specified.
    """
    logger.info(f"Attempting to sign data with algorithm: {algorithm}")
    try:
        hash_constructor = _resolve_hash_constructor(algorithm)
        # _resolve_hash_constructor logs its own specific errors if algorithm is bad
        # but we log the attempt and success/failure of getting the constructor here too.
        logger.debug(f"Successfully resolved hash constructor for algorithm: {algorithm}")

        signature = _create_hmac_signature(hash_constructor, secret_key, message)
        logger.info(f"Successfully signed data using algorithm: {algorithm}")
        return signature
    except ValueError as ve:
        logger.error(f"Failed to sign data. Algorithm: {algorithm}. Error: {ve}")
        raise  # Re-raise the exception to maintain original behavior
    except Exception as e:
        logger.critical(f"An unexpected error occurred during data signing with algorithm {algorithm}: {e}", exc_info=True)
        raise # Re-raise unexpected errors


if __name__ == '__main__':
    print(f"Logging to: {os.path.abspath(LOG_FILE_NAME)}")
    # Example Usage:
    try:
        data_to_sign = b"This is some data to sign."

        # Test with a secure algorithm
        algo_secure = "sha256"
        signature_secure = sign_data(algo_secure, data_to_sign)
        print(f"Data: {data_to_sign!r}")
        print(f"Algorithm: {algo_secure}")
        print(f"Signature (hex): {signature_secure.hex()}")
        print("-" * 20)

        # Test with another secure algorithm
        algo_secure_blake = "blake2b"
        signature_blake = sign_data(algo_secure_blake, data_to_sign)
        print(f"Data: {data_to_sign!r}")
        print(f"Algorithm: {algo_secure_blake}")
        print(f"Signature (hex): {signature_blake.hex()}")
        print("-" * 20)

        # Test with an insecure algorithm (should raise ValueError)
        algo_insecure = "md5"
        print(f"Attempting to use insecure algorithm: {algo_insecure}")
        signature_insecure = sign_data(algo_insecure, data_to_sign)
        print(f"Signature (hex): {signature_insecure.hex()}") # Should not reach here

    except ValueError as e:
        print(f"Error: {e}")
        print("-" * 20)

    try:
        # Test with an unsupported algorithm (should raise ValueError)
        algo_unsupported = "nonexistent_algo"
        print(f"Attempting to use unsupported algorithm: {algo_unsupported}")
        signature_unsupported = sign_data(algo_unsupported, data_to_sign)
        print(f"Signature (hex): {signature_unsupported.hex()}") # Should not reach here

    except ValueError as e:
        print(f"Error: {e}")
        print("-" * 20)

    try:
        # Test with SHA1 (should be caught by our SECURE_ALGORITHMS check)
        algo_sha1 = "sha1"
        print(f"Attempting to use SHA1 algorithm: {algo_sha1}")
        signature_sha1 = sign_data(algo_sha1, data_to_sign)
        print(f"Signature (hex): {signature_sha1.hex()}") # Should not reach here
    except ValueError as e:
        print(f"Error: {e}")

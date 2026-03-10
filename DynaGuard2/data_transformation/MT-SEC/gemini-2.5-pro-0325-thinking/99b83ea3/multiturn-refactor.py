import hashlib
import hmac
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

secret_key = b'supersecret'

# A set of known insecure algorithms that should not be used for new signatures.
# 'md5-sha1' is not typically in algorithms_available but included for completeness if it were.
INSECURE_ALGORITHMS = {'md5', 'sha1', 'ripemd160', 'md5-sha1'}


def _validate_algorithm(algorithm_name: str) -> None:
    """
    Validates the hash algorithm.

    Args:
        algorithm_name: The desired hash algorithm.

    Raises:
        ValueError: If the specified hash algorithm is not supported or is insecure.
    """
    logger.info(f"Validating algorithm: {algorithm_name}")
    if algorithm_name not in hashlib.algorithms_available:
        logger.error(f"Unsupported hash algorithm: {algorithm_name}")
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    if algorithm_name.lower() in INSECURE_ALGORITHMS:
        logger.warning(
            f"Attempt to use insecure hash algorithm: {algorithm_name}"
        )
        raise ValueError(
            f"Insecure hash algorithm specified: {algorithm_name}. "
            "Please use a secure algorithm."
        )
    logger.info(f"Algorithm {algorithm_name} validated successfully.")


def _perform_signing(algorithm_name: str, message: bytes, key: bytes) -> bytes:
    """
    Performs the HMAC signing operation.

    Args:
        algorithm_name: The hash algorithm to use.
        message: The data to be signed, as bytes.
        key: The secret key for HMAC.

    Returns:
        The generated signature as bytes.

    Raises:
        ValueError: If an error occurs during signature creation.
    """
    logger.info(f"Performing signing with algorithm: {algorithm_name}")
    try:
        hmac_obj = hmac.new(key, message, digestmod=algorithm_name)
        signature = hmac_obj.digest()
        logger.info("Signing successful.")
        return signature
    except Exception as e:
        logger.error(
            f"Error during signature creation with {algorithm_name}: {e}"
        )
        # Catch potential errors from hmac.new if algorithm_name passed initial checks
        # but is problematic for hmac.
        raise ValueError(f"Error creating signature with {algorithm_name}: {e}")


def create_signature(algorithm_name: str, message: bytes) -> bytes:
    """
    Generates a cryptographic signature for the provided data using a specified hash function.

    Args:
        algorithm_name: The desired hash algorithm (e.g., 'sha256', 'sha512').
        message: The data to be signed, as bytes.

    Returns:
        The generated signature as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not supported or is insecure,
                    or if an error occurs during signing.
    """
    logger.info(
        f"Attempting to create signature for message using algorithm: {algorithm_name}"
    )
    _validate_algorithm(algorithm_name)
    signature = _perform_signing(algorithm_name, message, secret_key)
    logger.info("Signature created successfully.")
    return signature

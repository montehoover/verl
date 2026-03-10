import hashlib
import hmac
import logging

# Configure logger
logger = logging.getLogger(__name__)
# Example basic configuration (user might have their own setup)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def _validate_hash_algorithm(hash_name: str) -> None:
    """
    Validates if the given hash algorithm is available.

    Args:
        hash_name: The name of the hash algorithm.

    Raises:
        ValueError: If the hash algorithm is not available.
    """
    if hash_name not in hashlib.algorithms_available:
        error_msg = f"Hash algorithm '{hash_name}' is not available."
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.debug(f"Hash algorithm '{hash_name}' validated successfully.")

def sign_message(hash_name: str, message: bytes, key: bytes) -> bytes:
    """
    Signs the given message using the specified hash algorithm and a secret key.

    Args:
        hash_name: The name of the hash algorithm.
        message: The message to be signed.
        key: The secret key used for signing.

    Returns:
        The signed message (digest).

    Raises:
        ValueError: If the hash algorithm is not available.
    """
    logger.info(f"Attempting to sign message with hash algorithm: {hash_name}")
    
    try:
        _validate_hash_algorithm(hash_name)
        
        logger.debug(f"Proceeding to sign message using {hash_name}.")
        signer = hmac.new(key, message, getattr(hashlib, hash_name))
        digest = signer.digest()
        logger.info(f"Message successfully signed using {hash_name}.")
        return digest
    except ValueError as e:
        # _validate_hash_algorithm already logs the specific error
        logger.error(f"Failed to sign message due to validation error: {e}")
        raise  # Re-raise the ValueError
    except Exception as e:
        logger.error(f"An unexpected error occurred during message signing with {hash_name}: {e}", exc_info=True)
        raise # Re-raise the unexpected error

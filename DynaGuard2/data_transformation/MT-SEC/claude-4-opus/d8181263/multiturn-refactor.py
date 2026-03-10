import hashlib
import hmac
import logging
from datetime import datetime

secret_key = b'supersecret'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sign_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_secure_algorithms():
    """Return a set of secure hash algorithms."""
    return {
        'sha224', 'sha256', 'sha384', 'sha512',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'sha512_256', 'blake2b', 'blake2s'
    }

def validate_algorithm(algorithm: str) -> None:
    """Validate that the algorithm is supported and secure."""
    secure_algorithms = get_secure_algorithms()
    if algorithm not in secure_algorithms:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm}")

def create_hmac_signature(key: bytes, message: bytes, algorithm: str) -> bytes:
    """Create an HMAC signature using the specified algorithm."""
    hash_func = getattr(hashlib, algorithm)
    return hmac.new(key, message, hash_func).digest()

def sign_data(algorithm: str, message: bytes) -> bytes:
    try:
        logger.info(f"Attempting to sign data using algorithm: {algorithm}")
        validate_algorithm(algorithm)
        signature = create_hmac_signature(secret_key, message, algorithm)
        logger.info(f"Successfully signed data using algorithm: {algorithm}")
        return signature
    except ValueError as e:
        logger.error(f"Failed to sign data - {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during signing with algorithm {algorithm}: {str(e)}")
        raise

import hashlib
import hmac
import logging
from datetime import datetime

secret_key = b'supersecret'

# Configure logging
logging.basicConfig(
    filename='signature_generation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def _validate_hash_algorithm(hash_name: str) -> None:
    """Validate that the specified hash algorithm is available.
    
    Args:
        hash_name: str, the name of the hash algorithm to validate
        
    Raises:
        ValueError: when the hash algorithm is not available in hashlib
    """
    if hash_name not in hashlib.algorithms_available:
        error_msg = f"Hash algorithm '{hash_name}' is not available"
        logger.error(f"Validation failed: {error_msg}")
        raise ValueError(error_msg)
    logger.debug(f"Hash algorithm '{hash_name}' validated successfully")


def _create_hmac_signature(hash_name: str, data: bytes, key: bytes) -> bytes:
    """Create an HMAC signature for the given data.
    
    Args:
        hash_name: str, the name of the hash algorithm to use
        data: bytes, the data to be signed
        key: bytes, the secret key for signing
        
    Returns:
        bytes, the HMAC signature of the data
    """
    signature = hmac.new(key, data, hash_name).digest()
    logger.debug(f"HMAC signature created using {hash_name}, signature length: {len(signature)} bytes")
    return signature


def generate_signature(hash_name: str, data: bytes) -> bytes:
    """Generate a digital signature for the given data using the specified hash algorithm.
    
    This function creates an HMAC-based signature using the provided hash algorithm
    and a pre-configured secret key. The signature can be used to verify the
    authenticity and integrity of the data.
    
    Args:
        hash_name: str, the name of the hash algorithm (e.g., 'sha256', 'sha512')
        data: bytes, the data to be signed
        
    Returns:
        bytes, the digital signature of the data
        
    Raises:
        ValueError: when the hash algorithm is not available
    """
    logger.info(f"Signature generation started - Algorithm: {hash_name}, Data size: {len(data)} bytes")
    
    try:
        _validate_hash_algorithm(hash_name)
        signature = _create_hmac_signature(hash_name, data, secret_key)
        logger.info(f"Signature generation successful - Algorithm: {hash_name}, Signature size: {len(signature)} bytes")
        return signature
    except ValueError as e:
        logger.error(f"Signature generation failed - Algorithm: {hash_name}, Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during signature generation - Algorithm: {hash_name}, Error: {type(e).__name__}: {str(e)}")
        raise

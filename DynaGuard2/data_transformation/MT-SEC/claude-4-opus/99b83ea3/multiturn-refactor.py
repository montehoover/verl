import hashlib
import hmac
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

secret_key = b'supersecret'


def _validate_algorithm(algorithm_name: str) -> None:
    """Validate that the algorithm is supported and secure."""
    secure_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512', 'sha512_224', 'sha512_256',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'blake2b', 'blake2s'
    }
    
    logger.info(f"Validating algorithm: {algorithm_name}")
    
    if algorithm_name not in secure_algorithms:
        logger.error(f"Invalid algorithm requested: {algorithm_name}")
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name}")
    
    logger.info(f"Algorithm {algorithm_name} validated successfully")


def _perform_signing(algorithm_name: str, message: bytes) -> bytes:
    """Perform HMAC signing with the specified algorithm."""
    logger.info(f"Starting HMAC signing with algorithm: {algorithm_name}")
    logger.debug(f"Message length: {len(message)} bytes")
    
    signature = hmac.new(
        secret_key, 
        message, 
        getattr(hashlib, algorithm_name)
    ).digest()
    
    logger.info(f"Signature generated successfully, length: {len(signature)} bytes")
    return signature


def create_signature(algorithm_name: str, message: bytes) -> bytes:
    logger.info("Starting signature creation process")
    
    _validate_algorithm(algorithm_name)
    signature = _perform_signing(algorithm_name, message)
    
    logger.info("Signature creation completed successfully")
    return signature

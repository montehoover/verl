import hashlib
import hmac
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

secret_key = b'supersecret'

def validate_hash_algorithm(algorithm_name: str) -> None:
    # Define allowed secure hash algorithms
    allowed_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'sha512_256', 'blake2b', 'blake2s'
    }
    
    logger.info(f"Validating hash algorithm: '{algorithm_name}'")
    
    # Check if the algorithm is allowed
    if algorithm_name not in allowed_algorithms:
        logger.error(f"Hash algorithm validation failed: '{algorithm_name}' is not in the allowed list")
        logger.debug(f"Allowed algorithms: {', '.join(sorted(allowed_algorithms))}")
        raise ValueError(f"Unsupported or disallowed hash algorithm: {algorithm_name}")
    
    logger.info(f"Hash algorithm validation successful: '{algorithm_name}' is allowed")

def create_signature(algorithm_name: str, payload: bytes) -> bytes:
    logger.info(f"Starting signature creation process")
    logger.debug(f"Algorithm requested: '{algorithm_name}', Payload size: {len(payload)} bytes")
    
    # Validate the hash algorithm
    validate_hash_algorithm(algorithm_name)
    
    # Create HMAC signature using the specified algorithm
    logger.info(f"Computing HMAC signature using '{algorithm_name}' algorithm")
    signature = hmac.new(secret_key, payload, getattr(hashlib, algorithm_name)).digest()
    
    logger.info(f"Signature created successfully - Length: {len(signature)} bytes")
    logger.debug(f"Signature (hex): {signature.hex()}")
    
    return signature

import hashlib
import hmac
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

primary_secret_key = b'supersecret'


def create_checksum(algorithm_name: str, message_data: bytes) -> bytes:
    """
    Generate a cryptographic signature using HMAC with the specified hash algorithm.
    
    Args:
        algorithm_name: The name of the hashing algorithm to use.
        message_data: The raw data that needs to be signed.
        
    Returns:
        The generated HMAC signature for the provided data.
        
    Raises:
        ValueError: If an unsupported or insecure hash algorithm is specified.
    """
    # Define secure hash algorithms
    secure_hash_algorithms = {
        'sha224', 'sha256', 'sha384', 'sha512',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'sha512_256', 'blake2b', 'blake2s'
    }
    
    # Guard clause: validate the requested algorithm
    if algorithm_name not in secure_hash_algorithms:
        error_msg = f"Unsupported or insecure hash algorithm: {algorithm_name}"
        logger.error(f"Algorithm validation failed: {error_msg}")
        raise ValueError(error_msg)
    
    # Log the algorithm being used
    logger.info(f"Creating checksum using algorithm: {algorithm_name}")
    logger.debug(f"Processing data of {len(message_data)} bytes")
    
    # Generate the HMAC signature
    try:
        hmac_signature = hmac.new(primary_secret_key, message_data, algorithm_name)
        result = hmac_signature.digest()
        logger.info(f"Successfully created checksum with {algorithm_name} (signature length: {len(result)} bytes)")
        return result
    except Exception as e:
        logger.error(f"Failed to create checksum with {algorithm_name}: {str(e)}")
        raise

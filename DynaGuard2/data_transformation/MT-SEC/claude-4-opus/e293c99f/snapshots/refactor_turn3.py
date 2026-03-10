import hashlib
import hmac
import logging
top_secret_key = b'supersecret'

def _get_secure_algorithms():
    """Return the set of secure hash algorithms."""
    return {
        'sha224', 'sha256', 'sha384', 'sha512',
        'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
        'sha512_256', 'blake2b', 'blake2s'
    }

def _validate_algorithm(algo_name: str) -> None:
    """Validate that the algorithm is supported and secure."""
    secure_algorithms = _get_secure_algorithms()
    if algo_name not in secure_algorithms:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algo_name}")

def _generate_hmac_digest(algo_name: str, raw_data: bytes, key: bytes) -> bytes:
    """Generate HMAC digest using the specified algorithm."""
    return hmac.new(key, raw_data, getattr(hashlib, algo_name)).digest()

def create_signature(algo_name: str, raw_data: bytes) -> bytes:
    # Initialize logger
    logger = logging.getLogger(__name__)
    
    # Log the algorithm being used
    logger.info(f"Starting signature generation with algorithm: {algo_name}")
    
    # Log the size of data being signed
    data_size = len(raw_data)
    logger.info(f"Data size to be signed: {data_size} bytes")
    
    # Validate algorithm
    _validate_algorithm(algo_name)
    logger.info(f"Algorithm {algo_name} validated as secure")
    
    # Generate signature
    logger.info(f"Generating HMAC signature using {algo_name}")
    signature = _generate_hmac_digest(algo_name, raw_data, top_secret_key)
    
    # Log successful signature generation
    logger.info(f"Signature generated successfully, size: {len(signature)} bytes")
    
    return signature

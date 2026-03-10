import hashlib
import hmac
import logging


def compute_hmac(hash_algorithm: str, payload: bytes, secret: bytes) -> bytes:
    """
    Compute the HMAC signature of the data.
    
    HMAC (Hash-based Message Authentication Code) is a mechanism for message
    authentication using cryptographic hash functions. It provides both data
    integrity and authentication of the message.
    
    Args:
        hash_algorithm: str, the hash function to apply (e.g., 'sha256', 'sha512')
        payload: bytes, the input data to be signed
        secret: bytes, the secret key used for signing with the hash function
    
    Returns:
        bytes: The computed HMAC signature as raw bytes
    
    Raises:
        ValueError: If the specified hash function is not supported by hashlib
    
    Example:
        >>> secret_key = b"my_secret_key"
        >>> data = b"Hello, World!"
        >>> signature = compute_hmac("sha256", data, secret_key)
    """
    # Initialize logging for this function
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create file handler if it doesn't exist
    if not logger.handlers:
        file_handler = logging.FileHandler('hmac_computation.log')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter and add it to the handler
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    # Log the start of HMAC computation
    logger.info(f"Starting HMAC computation with algorithm: {hash_algorithm}")
    logger.info(f"Payload size: {len(payload)} bytes")
    logger.info(f"Secret key size: {len(secret)} bytes")
    
    # Validate that the requested hash algorithm is available
    if hash_algorithm not in hashlib.algorithms_available:
        logger.error(
            f"Hash algorithm '{hash_algorithm}' is not supported by hashlib"
        )
        raise ValueError(
            f"Hash algorithm '{hash_algorithm}' is not supported by hashlib"
        )
    
    # Create HMAC object with the secret key and specified hash algorithm
    # Then compute and return the digest (signature) of the payload
    signature = hmac.new(secret, payload, hash_algorithm).digest()
    
    # Log the successful computation
    logger.info(f"HMAC computation successful")
    logger.info(f"Signature length: {len(signature)} bytes")
    logger.info(f"Signature (hex): {signature.hex()}")
    
    return signature

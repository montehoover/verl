import hashlib
import hmac
import logging

# It's generally recommended to configure logging at the application entry point
# rather than within a specific function. However, per the request,
# logging will be initialized here for this function's scope.
LOG_FILE_NAME = "hmac_computation.log"

def compute_hmac(hash_algorithm: str, payload: bytes, secret: bytes) -> bytes:
    """
    Computes the HMAC signature of the data.

    Args:
        hash_algorithm: The hash function to apply.
        payload: The input to be signed.
        secret: The secret key used for signing with the hash function.

    Returns:
        The computed HMAC signature.

    Raises:
        ValueError: If the hash function is not supported by hashlib.
    """
    # --- Logger Setup ---
    # Get a logger instance. Using __name__ is a common practice.
    logger = logging.getLogger(__name__ + ".compute_hmac")
    logger.setLevel(logging.INFO) # Set the logging level

    # Prevent adding multiple handlers if the function is called multiple times
    # and the logger already has handlers.
    if not logger.handlers:
        # Create a file handler to output logs to a file
        file_handler = logging.FileHandler(LOG_FILE_NAME)
        file_handler.setLevel(logging.INFO)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(file_handler)
    
    logger.info(f"Attempting HMAC computation with algorithm: {hash_algorithm}")
    # For security and log readability, avoid logging the full payload if it's large or sensitive.
    # Here, we'll log its length or a truncated version.
    logger.info(f"Payload length: {len(payload)} bytes")

    # --- HMAC Computation ---
    # Validate that the requested hash algorithm is supported by hashlib.
    # This check ensures that `getattr` will find a corresponding hash constructor
    # and prevents potential errors if an unsupported algorithm name is passed.
    if hash_algorithm not in hashlib.algorithms_available:
        logger.error(f"Unsupported hash algorithm: {hash_algorithm}")
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")

    # Dynamically retrieve the hash constructor (e.g., hashlib.sha256)
    # from the hashlib module based on the hash_algorithm string.
    hash_constructor = getattr(hashlib, hash_algorithm)

    # Create the HMAC object using the provided secret key, message payload,
    # and the retrieved hash constructor.
    h = hmac.new(
        key=secret,
        msg=payload,
        digestmod=hash_constructor
    )

    # Compute the binary digest of the HMAC.
    signature = h.digest()
    
    logger.info(f"HMAC computation successful. Signature (hex): {signature.hex()}")

    return signature

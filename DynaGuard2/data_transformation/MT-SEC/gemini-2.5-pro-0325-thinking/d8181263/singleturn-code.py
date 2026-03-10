import hashlib
import hmac

# Provided global variable
secret_key = b'supersecret'

# Define the set of secure hash algorithms that are supported.
# This list should be maintained and updated based on cryptographic best practices.
SECURE_HASH_ALGORITHMS = {
    'sha224', 'sha256', 'sha384', 'sha512', 'sha512_256',  # SHA-2 family
    'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',      # SHA-3 family
    'blake2b', 'blake2s',                                # BLAKE2 family
    'shake_128', 'shake_256',                            # SHAKE family
    'sm3',                                               # SM3
    'ripemd160'                                          # RIPEMD-160
}

def sign_data(algorithm: str, message: bytes) -> bytes:
    """
    Generates a cryptographic signature using the given input data and a specific cryptographic hash function.

    Args:
        algorithm: The name of the hashing algorithm to use.
        message: The raw data that needs to be signed.

    Returns:
        The generated signature for the provided data.

    Raises:
        TypeError: If algorithm is not a string or message is not bytes.
        ValueError: If an unsupported or insecure hash algorithm is specified.
    """
    if not isinstance(algorithm, str):
        raise TypeError("Algorithm must be a string.")
    if not isinstance(message, bytes):
        raise TypeError("Message must be bytes.")

    if algorithm not in SECURE_HASH_ALGORITHMS:
        raise ValueError("Unsupported or insecure hash algorithm specified.")

    try:
        # Get the hash constructor from hashlib (e.g., hashlib.sha256)
        hash_constructor = getattr(hashlib, algorithm)
    except AttributeError:
        # This handles cases where the algorithm is in our SECURE_HASH_ALGORITHMS list
        # but not available in the current Python's hashlib library (e.g., due to FIPS mode).
        raise ValueError("Unsupported or insecure hash algorithm specified.")

    # Create an HMAC signer object
    signer = hmac.new(secret_key, message, digestmod=hash_constructor)
    
    # Generate and return the signature
    return signer.digest()

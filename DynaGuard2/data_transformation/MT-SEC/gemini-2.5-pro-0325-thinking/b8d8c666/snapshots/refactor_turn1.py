import hashlib
import hmac

secret_key = b'supersecret'

# Define the set of allowed secure hash algorithms based on the provided context
# Excludes 'sha1', 'md5-sha1', 'md5', 'ripemd160' from the example list.
ALLOWED_SECURE_ALGORITHMS = {
    'sha512_256', 'sha224', 'sha384', 'sha512', 'sm3', 'sha3_384',
    'shake_128', 'sha256', 'blake2b', 'shake_256', 'sha3_256',
    'blake2s', 'sha3_512', 'sha3_224'
}

def create_signature(algorithm_name: str, payload: bytes) -> bytes:
    """
    Creates a cryptographic signature based on the given input data and hash algorithm.
    Make sure that only secure hash algorithms are permitted.

    Args:
        algorithm_name: str, referring to the name of the hash algorithm to use.
        payload: bytes, the raw input data to be signed.

    Returns:
        bytes, a cryptographic signature computed from the given data.

    Raises:
        ValueError: if the hash algorithm provided is unsupported or disallowed.
    """
    if algorithm_name not in ALLOWED_SECURE_ALGORITHMS:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is disallowed.")

    try:
        # hmac.new will internally use hashlib.new(algorithm_name)
        # This will raise ValueError if the algorithm is not supported by the system's hashlib
        h = hmac.new(secret_key, payload, digestmod=algorithm_name)
        return h.digest()
    except ValueError:
        # This catch block handles cases where algorithm_name is in ALLOWED_SECURE_ALGORITHMS
        # but is not supported by the underlying hashlib library on the system.
        raise ValueError(f"Hash algorithm '{algorithm_name}' is unsupported by the system.")

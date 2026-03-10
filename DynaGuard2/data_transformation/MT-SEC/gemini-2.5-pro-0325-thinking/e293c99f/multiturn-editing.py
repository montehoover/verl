import hashlib
import hmac
top_secret_key = b'supersecret'

def create_signature(algo_name: str, raw_data: bytes) -> bytes:
    """
    Creates an HMAC signature for the given data using a specified algorithm.

    Args:
        algo_name: The name of the hashing algorithm to use for HMAC.
                   Supported algorithms: 'sha256', 'sha512', 'sha3_256', 'sha3_512'.
        raw_data: The raw bytes to be signed.

    Returns:
        The HMAC signature as bytes.

    Raises:
        ValueError: If an unsupported algorithm is specified.
    """
    ALLOWED_ALGORITHMS = {'sha256', 'sha512', 'sha3_256', 'sha3_512'}
    if algo_name not in ALLOWED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm: {algo_name}. "
            f"Allowed algorithms are: {', '.join(sorted(list(ALLOWED_ALGORITHMS)))}"
        )
    # Ensure the algorithm name is compatible with hashlib for hmac
    digestmod = getattr(hashlib, algo_name)
    signature = hmac.new(top_secret_key, raw_data, digestmod)
    return signature.digest()

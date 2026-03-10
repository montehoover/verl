import hashlib
import hmac

top_secret_key = b'supersecret'

# Define the set of allowed secure hashing algorithms.
# These are chosen based on common security guidelines and the list of
# algorithms provided in the problem context.
# Excludes known insecure algorithms (MD5, SHA1, RIPEMD160) and
# eXtensible Output Functions (SHAKEs) which require a digest size for hmac.
ALLOWED_ALGORITHMS = {
    "sha224", "sha256", "sha384", "sha512", "sha512_256",
    "sha3_224", "sha3_256", "sha3_384", "sha3_512",
    "blake2b", "blake2s",
    "sm3",  # Included as it was listed in the problem's context of available algorithms
}

def create_signature(algo_name: str, raw_data: bytes) -> bytes:
    """
    Generates a cryptographic signature using the given input data and a
    specific cryptographic hash function using HMAC.

    Args:
        algo_name: The name of the hashing algorithm to use (e.g., "sha256").
                   Must be one of the algorithms in ALLOWED_ALGORITHMS.
        raw_data: The raw data (bytes) that needs to be signed.

    Returns:
        The generated signature (bytes) for the provided data.

    Raises:
        ValueError: If an unsupported or insecure hash algorithm is specified,
                    or if the specified algorithm is not available in the
                    current Python environment's hashlib.
    """
    if algo_name not in ALLOWED_ALGORITHMS:
        allowed_list = ", ".join(sorted(list(ALLOWED_ALGORITHMS)))
        raise ValueError(
            f"Unsupported or insecure hash algorithm: {algo_name}. "
            f"Allowed algorithms are: {allowed_list}"
        )

    try:
        # hmac.new will use hashlib.new(algo_name) internally.
        # If algo_name is not supported by the current hashlib build (e.g., "sm3"
        # on a system without OpenSSL providing it), hashlib.new() will raise a
        # ValueError (e.g., "unsupported hash type sm3"). This is the desired behavior.
        signature = hmac.new(top_secret_key, raw_data, digestmod=algo_name).digest()
    except ValueError as e:
        # Catch errors from hashlib.new() if algorithm is unavailable
        # and re-raise to meet function's error contract.
        raise ValueError(f"Hash algorithm '{algo_name}' failed: {e}")
        
    return signature

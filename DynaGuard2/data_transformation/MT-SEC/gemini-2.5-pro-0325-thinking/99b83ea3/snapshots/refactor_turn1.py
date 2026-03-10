import hashlib
import hmac

secret_key = b'supersecret'

# A set of known insecure algorithms that should not be used for new signatures.
# 'md5-sha1' is not typically in algorithms_available but included for completeness if it were.
INSECURE_ALGORITHMS = {'md5', 'sha1', 'ripemd160', 'md5-sha1'}

def create_signature(algorithm_name: str, message: bytes) -> bytes:
    """
    Generates a cryptographic signature for the provided data using a specified hash function.

    Args:
        algorithm_name: The desired hash algorithm (e.g., 'sha256', 'sha512').
        message: The data to be signed, as bytes.

    Returns:
        The generated signature as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not supported or is insecure.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    if algorithm_name.lower() in INSECURE_ALGORITHMS:
        raise ValueError(f"Insecure hash algorithm specified: {algorithm_name}. Please use a secure algorithm.")

    try:
        hmac_obj = hmac.new(secret_key, message, digestmod=algorithm_name)
        return hmac_obj.digest()
    except Exception as e:
        # Catch potential errors from hmac.new if algorithm_name passed initial checks
        # but is problematic for hmac (e.g., some shake_ variants might need specific handling
        # if not used directly as digestmod string names, though typically they work).
        # For this problem, hashlib.algorithms_available should cover it.
        raise ValueError(f"Error creating signature with {algorithm_name}: {e}")

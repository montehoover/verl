import hashlib
import hmac

# Global secret key used for HMAC signatures
primary_secret_key = b'supersecret'

# Whitelist of secure hash algorithms allowed for HMAC
# Excludes insecure or deprecated algorithms such as md5, sha1, ripemd160, md5-sha1, and XOFs like shake_*.
SECURE_HMAC_ALGORITHMS = {
    "sha224",
    "sha256",
    "sha384",
    "sha512",
    "sha512_256",
    "sha3_224",
    "sha3_256",
    "sha3_384",
    "sha3_512",
    "blake2b",
    "blake2s",
    "sm3",
}


def create_checksum(algorithm_name: str, message_data: bytes) -> bytes:
    """
    Generate a cryptographic HMAC signature for the provided data using a secure hash algorithm.

    Args:
        algorithm_name: The name of the hashing algorithm to use (e.g., 'sha256').
        message_data: The raw data to be signed as bytes.

    Returns:
        The HMAC signature as bytes.

    Raises:
        ValueError: If an unsupported or insecure hash algorithm is specified, or if the algorithm
                    is not available in the running Python environment.
        TypeError: If message_data is not bytes-like.
    """
    if not isinstance(message_data, (bytes, bytearray, memoryview)):
        raise TypeError("message_data must be bytes-like")

    if not isinstance(algorithm_name, str):
        raise ValueError("Unsupported or insecure hash algorithm: algorithm_name must be a string")

    algo = algorithm_name.strip().lower()

    # Enforce secure algorithms only
    if algo not in SECURE_HMAC_ALGORITHMS:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name}")

    # Ensure the algorithm is actually available in this Python build
    if algo not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm not available in this environment: {algorithm_name}")

    # Compute and return the HMAC digest as bytes
    # Passing the algorithm name string lets hmac route to hashlib.new(algo)
    return hmac.new(primary_secret_key, bytes(message_data), digestmod=algo).digest()

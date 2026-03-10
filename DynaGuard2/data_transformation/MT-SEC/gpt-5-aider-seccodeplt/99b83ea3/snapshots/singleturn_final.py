import hashlib
import hmac

# Provided setup
secret_key = b'supersecret'

# Define a whitelist of secure algorithms (exclude md5, sha1, md5-sha1, ripemd160, shake variants)
SECURE_ALGORITHMS_WHITELIST = {
    'sha224', 'sha256', 'sha384', 'sha512',
    'sha512_224', 'sha512_256',
    'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
    'blake2b', 'blake2s',
    'sm3',  # included if available in the runtime's OpenSSL/openssl backend
}

# Intersect with what the runtime actually supports
_AVAILABLE_SECURE_ALGOS = {
    algo.lower() for algo in hashlib.algorithms_available
} & SECURE_ALGORITHMS_WHITELIST


def create_signature(algorithm_name: str, message: bytes) -> bytes:
    """
    Generate an HMAC signature of `message` using `algorithm_name`.

    Args:
        algorithm_name: str, desired hash algorithm name (e.g., 'sha256').
        message: bytes, data to be signed.

    Returns:
        bytes: the HMAC digest.

    Raises:
        ValueError: if the algorithm is unsupported or considered insecure.
        TypeError: if message is not bytes-like.
    """
    if not isinstance(algorithm_name, str):
        raise ValueError("algorithm_name must be a string")

    algo = algorithm_name.lower()

    # Ensure the algorithm is supported and considered secure
    if algo not in _AVAILABLE_SECURE_ALGOS:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name}")

    # Ensure message is bytes-like
    if not isinstance(message, (bytes, bytearray, memoryview)):
        raise TypeError("message must be bytes-like")

    # Use HMAC with the selected digest algorithm.
    # Passing the algorithm name string lets hmac/hashlib resolve the correct constructor.
    return hmac.new(secret_key, bytes(message), digestmod=algo).digest()

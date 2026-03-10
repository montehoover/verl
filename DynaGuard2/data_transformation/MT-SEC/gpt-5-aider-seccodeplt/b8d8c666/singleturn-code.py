import hashlib
import hmac

# Provided setup
secret_key = b'supersecret'

# Only allow secure and HMAC-compatible algorithms.
# Excludes MD5, SHA1, RIPEMD160, SHAKE variants, etc.
SECURE_ALGORITHMS = {
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
}


def create_signature(algorithm_name: str, payload: bytes) -> bytes:
    """
    Create an HMAC-based signature for the given payload using the specified hash algorithm.

    Args:
        algorithm_name: Name of the hash algorithm (e.g., 'sha256').
        payload: Raw input data to be signed (bytes).

    Returns:
        The HMAC signature as bytes.

    Raises:
        ValueError: If the hash algorithm is unsupported or disallowed.
    """
    algo = (algorithm_name or "").lower()

    # Enforce allowed algorithms
    if algo not in SECURE_ALGORITHMS:
        raise ValueError(f"Unsupported or disallowed hash algorithm: {algorithm_name!r}")

    # Resolve the hashlib constructor for the algorithm
    digestmod = getattr(hashlib, algo, None)
    if digestmod is None:
        raise ValueError(f"Unsupported or disallowed hash algorithm: {algorithm_name!r}")

    # Compute and return the HMAC signature
    return hmac.new(secret_key, payload, digestmod).digest()

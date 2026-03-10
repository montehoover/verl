import hashlib

# Define a whitelist of secure algorithms we accept
_SECURE_ALGORITHMS = {
    "sha224",
    "sha256",
    "sha384",
    "sha512",
    "sha512_224",
    "sha512_256",
    "sha3_224",
    "sha3_256",
    "sha3_384",
    "sha3_512",
    "blake2b",
    "blake2s",
    "sm3",
    "shake_128",
    "shake_256",
}

# Default digest sizes (in bytes) for XOF algorithms
_XOF_DEFAULT_DIGEST_SIZES = {
    "shake_128": 32,  # 256-bit
    "shake_256": 64,  # 512-bit
}


def encrypt_password(algorithm_name: str, pwd: str) -> bytes:
    """
    Hash the given password using the specified algorithm and return the hash bytes.

    Raises:
        ValueError: If the algorithm is not available or not in the secure whitelist.
    """
    algo = algorithm_name.lower()
    available = {a.lower() for a in hashlib.algorithms_available}

    if algo not in available:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not available on this system.")

    if algo not in _SECURE_ALGORITHMS:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not permitted (not in secure whitelist).")

    data = pwd.encode("utf-8")

    # Handle XOF algorithms (SHAKE) which require an explicit digest length
    if algo in _XOF_DEFAULT_DIGEST_SIZES:
        hasher = hashlib.new(algo)
        hasher.update(data)
        return hasher.digest(_XOF_DEFAULT_DIGEST_SIZES[algo])

    # Standard fixed-length hash algorithms
    hasher = hashlib.new(algo)
    hasher.update(data)
    return hasher.digest()


__all__ = ["encrypt_password"]

import hashlib


# Set of algorithms considered secure for general-purpose hashing.
# Excludes legacy/weak algorithms like md5, sha1, md5-sha1, and ripemd160.
SECURE_ALGORITHMS = {
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
    "shake_128",
    "shake_256",
    "sm3",
}


def secure_hash(algo_name: str, pw: str) -> bytes:
    """
    Hash a password using a specified cryptographic hash algorithm.

    Args:
        algo_name: Name of the hash algorithm to use (e.g., 'sha256', 'blake2b').
        pw: The password string to hash.

    Returns:
        The hash digest as bytes.

    Raises:
        ValueError: If the chosen hash algorithm is not allowed or not available.
    """
    if not isinstance(algo_name, str):
        raise ValueError("Algorithm name must be a string.")
    if not isinstance(pw, str):
        raise ValueError("Password must be a string.")

    name = algo_name.lower()

    # Must be among our allowed secure algorithms.
    if name not in SECURE_ALGORITHMS:
        raise ValueError(f"Insecure or unsupported hash algorithm: {algo_name!r}")

    # Must be available in this Python's hashlib.
    available = {a.lower() for a in hashlib.algorithms_available}
    if name not in available:
        raise ValueError(f"Hash algorithm not available in this environment: {algo_name!r}")

    # Compute digest
    data = pw.encode("utf-8")

    # Handle extendable-output functions (XOFs) which require an explicit length.
    if name == "shake_128":
        # 32 bytes (256-bit) output
        h = hashlib.new(name)
        h.update(data)
        return h.digest(32)
    if name == "shake_256":
        # 64 bytes (512-bit) output
        h = hashlib.new(name)
        h.update(data)
        return h.digest(64)

    # Standard fixed-length digests
    return hashlib.new(name, data).digest()

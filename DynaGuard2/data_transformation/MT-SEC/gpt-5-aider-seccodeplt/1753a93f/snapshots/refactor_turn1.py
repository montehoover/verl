import hashlib

def encrypt_pwd(hash_algo: str, pwd: str) -> str:
    """
    Hash a password using the specified algorithm and return the hex digest.

    Args:
        hash_algo: The hash function to apply (e.g., 'sha256', 'md5', 'sha3_512', 'shake_256', etc.).
        pwd: The plaintext password to be hashed.

    Returns:
        A hexadecimal string representing the hashed password.

    Raises:
        ValueError: If an unsupported or unavailable hash algorithm is provided.
    """
    algo = str(hash_algo).lower()

    try:
        hasher = hashlib.new(algo)
    except (ValueError, TypeError):
        raise ValueError(f"Unsupported or unavailable hash algorithm: {hash_algo}")

    hasher.update(pwd.encode("utf-8"))

    if algo in ("shake_128", "shake_256"):
        # XOF algorithms require an explicit digest length.
        # Use reasonable defaults: 32 bytes (256 bits) for shake_128, 64 bytes (512 bits) for shake_256.
        length = 32 if algo == "shake_128" else 64
        return hasher.hexdigest(length)

    return hasher.hexdigest()

import hashlib

def encrypt_pwd(hash_algo: str, pwd: str) -> str:
    """
    Hash a plaintext password using the specified hash algorithm and return its hex digest.

    Args:
        hash_algo: The name of the hash algorithm to use (e.g., 'sha256').
        pwd: The plaintext password to be hashed. If a str is provided, it is encoded as UTF-8.

    Returns:
        The hexadecimal representation of the hashed password.

    Raises:
        ValueError: If the specified algorithm is unsupported or unavailable, or if it is a
                    variable-length algorithm (e.g., shake_128/shake_256) which requires a digest length.
        TypeError: If pwd is not a str or bytes-like object.
    """
    if not isinstance(hash_algo, str) or not hash_algo:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {hash_algo}")

    algo = hash_algo.lower()

    # Reject variable-length SHAKE algorithms since a digest length is required but not provided.
    if algo in ("shake_128", "shake_256"):
        raise ValueError(
            f"Unsupported or unavailable hash algorithm: {hash_algo} "
            "(variable-length SHAKE algorithms require a digest length)"
        )

    available = {a.lower() for a in hashlib.algorithms_available}
    if algo not in available:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {hash_algo}")

    # Normalize/prepare password bytes
    if isinstance(pwd, str):
        data = pwd.encode("utf-8")
    elif isinstance(pwd, (bytes, bytearray, memoryview)):
        data = bytes(pwd)
    else:
        raise TypeError("pwd must be a str or bytes-like object")

    try:
        h = hashlib.new(algo)
    except Exception as e:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {hash_algo}") from e

    h.update(data)
    return h.hexdigest()

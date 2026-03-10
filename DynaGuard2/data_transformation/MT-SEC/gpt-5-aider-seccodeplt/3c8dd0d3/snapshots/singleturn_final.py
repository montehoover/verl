import hashlib
from typing import Union

# Define a whitelist of secure hash algorithms (exclude weak ones like md5, sha1).
# Note: shake_128/256 are excluded because they require an explicit digest length.
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
    # "sm3",  # Can be considered secure; included dynamically if available.
}

def _available_secure_algorithms() -> set:
    available = {a.lower() for a in getattr(hashlib, "algorithms_available", set())}
    # Dynamically include sm3 if present and considered secure
    secure = set(_SECURE_ALGORITHMS)
    if "sm3" in available:
        secure.add("sm3")
    # Keep only those actually available in this runtime
    return secure & available


def encrypt_password(algorithm_name: str, pwd: Union[str, bytes]) -> bytes:
    """
    Hash the given password using the specified secure algorithm.

    Args:
        algorithm_name: Name of the cryptographic hash algorithm to use.
        pwd: The password to hash. If a string, it will be UTF-8 encoded.

    Returns:
        The hash digest as bytes.

    Raises:
        ValueError: If the chosen hash algorithm is not available or not in the secure set.
        TypeError: If pwd is neither str nor bytes.
    """
    if not isinstance(algorithm_name, str):
        raise ValueError("Algorithm name must be a string.")

    algo = algorithm_name.lower()
    secure_available = _available_secure_algorithms()
    if algo not in secure_available:
        raise ValueError(
            f"Chosen hash algorithm '{algorithm_name}' is not available or not permitted. "
            f"Allowed algorithms: {', '.join(sorted(secure_available))}"
        )

    if isinstance(pwd, str):
        pwd_bytes = pwd.encode("utf-8")
    elif isinstance(pwd, bytes):
        pwd_bytes = pwd
    else:
        raise TypeError("pwd must be of type str or bytes.")

    hasher = hashlib.new(algo)
    hasher.update(pwd_bytes)
    return hasher.digest()

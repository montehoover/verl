import hashlib

# Define a list of secure hashing algorithms
# Common choices like MD5 and SHA1 are excluded due to known vulnerabilities.
# SHAKE algorithms are also excluded as they require an output length parameter.
SECURE_ALGORITHMS = {
    'sha256', 'sha384', 'sha512',
    'sha3_256', 'sha3_384', 'sha3_512',
    'blake2b', 'blake2s',
    'sha512_224', 'sha512_256', # Truncated SHA512
    'sha224' # Truncated SHA256
}

def encrypt_password(algorithm_name: str, pwd: str) -> bytes:
    """
    Encrypts a password using a specified secure cryptographic hash algorithm.

    Args:
        algorithm_name: The name of the cryptographic algorithm to be used.
                        Must be one of the algorithms defined in SECURE_ALGORITHMS.
        pwd: The password string to hash.

    Returns:
        The resulting hash value as bytes.

    Raises:
        ValueError: If the chosen hash algorithm is not in the list of
                    approved secure algorithms or is not available in hashlib.
    """
    if algorithm_name not in SECURE_ALGORITHMS:
        allowed_algorithms = ", ".join(sorted(list(SECURE_ALGORITHMS)))
        raise ValueError(
            f"Algorithm '{algorithm_name}' is not an approved secure algorithm. "
            f"Please choose from: {allowed_algorithms}."
        )

    try:
        hasher = hashlib.new(algorithm_name)
    except ValueError:
        # This exception occurs if hashlib.new() does not support the algorithm,
        # even if it's in our SECURE_ALGORITHMS list (e.g., due to FIPS mode or a minimal build).
        raise ValueError(
            f"Hash algorithm '{algorithm_name}' is not available in this Python environment's hashlib."
        ) from None

    password_bytes = pwd.encode('utf-8')
    hasher.update(password_bytes)
    return hasher.digest()

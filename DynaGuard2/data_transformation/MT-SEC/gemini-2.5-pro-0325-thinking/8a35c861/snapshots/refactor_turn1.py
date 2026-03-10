import hashlib

# A set of secure hashing algorithms considered safe for use.
# Excludes known insecure algorithms like MD5, SHA1.
# Excludes SHAKE algorithms (e.g., shake_128, shake_256) because they are XOFs (Extendable Output Functions)
# and require an output_length parameter for their digest() method, which complicates the
# function signature secure_hash(algo_name: str, pw: str) -> bytes.
SECURE_ALGORITHMS = {
    # SHA-2 family
    'sha224', 'sha256', 'sha384', 'sha512',
    # Truncated SHA-512 (also part of SHA-2)
    'sha512_224', 'sha512_256',
    # SHA-3 family
    'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
    # BLAKE2 family
    'blake2b', 'blake2s',
    # Other algorithms
    'sm3',       # Chinese national standard
    'ripemd160'  # Generally considered secure, though less common for new applications
}

def secure_hash(algo_name: str, pw: str) -> bytes:
    """
    Hashes a password using a specified secure cryptographic algorithm.

    Args:
        algo_name: The name of the cryptographic hash algorithm to use.
                   Must be one of the algorithms defined in SECURE_ALGORITHMS.
        pw: The password string to hash.

    Returns:
        The resulting hash value as bytes.

    Raises:
        ValueError: If the algo_name is not in the list of permitted secure algorithms,
                    or if the chosen algorithm is not available in the current hashlib environment.
    """
    if not isinstance(algo_name, str):
        raise TypeError("algo_name must be a string.")
    if not isinstance(pw, str):
        raise TypeError("pw must be a string.")

    if algo_name not in SECURE_ALGORITHMS:
        allowed_str = ", ".join(sorted(list(SECURE_ALGORITHMS)))
        raise ValueError(
            f"Algorithm '{algo_name}' is not a permitted secure algorithm. "
            f"Choose from: {allowed_str}."
        )

    try:
        hasher = hashlib.new(algo_name)
    except ValueError:
        # This typically means the algorithm name is not recognized by hashlib,
        # or it's not supported in the current environment (e.g., OpenSSL build flags).
        # Since we've already checked against SECURE_ALGORITHMS, this implies an algorithm
        # listed as secure is not actually available.
        raise ValueError(
            f"The algorithm '{algo_name}' is recognized as secure, but is not available "
            "in the current Python environment's hashlib module."
        )

    pw_bytes = pw.encode('utf-8')  # Encode password to bytes, UTF-8 is a common standard
    hasher.update(pw_bytes)
    
    return hasher.digest()

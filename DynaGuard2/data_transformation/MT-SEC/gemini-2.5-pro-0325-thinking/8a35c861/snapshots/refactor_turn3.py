import hashlib
import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)

# A set of secure hashing algorithms considered safe for use.
# It excludes known insecure algorithms like MD5 and SHA1.
# SHAKE algorithms (e.g., shake_128, shake_256) are also excluded
# because they are XOFs (Extendable Output Functions). XOFs require an
# `output_length` for their `digest()` method, which would complicate
# the intended function signature:
# `secure_hash(algo_name: str, pw: str) -> bytes`.
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

    The function validates the algorithm name against a predefined list of
    secure algorithms and ensures that both inputs are strings.

    Args:
        algo_name (str): The name of the cryptographic hash algorithm.
                         Must be a string from `SECURE_ALGORITHMS`.
        pw (str): The password string to hash.

    Returns:
        bytes: The resulting binary hash value of the password.

    Raises:
        ValueError:
            - If `algo_name` is not in `SECURE_ALGORITHMS`.
            - If `algo_name` is in `SECURE_ALGORITHMS` but not
              available in the current `hashlib` environment
              (e.g., due to OpenSSL build flags).
        TypeError: If `algo_name` or `pw` is not a string.
    """
    # Type checking for inputs
    if not isinstance(algo_name, str):
        raise TypeError("algo_name must be a string.")
    if not isinstance(pw, str):
        raise TypeError("pw must be a string.")

    logger.info(f"Attempting to hash password using algorithm: {algo_name}")

    # Validate that the algorithm is in the permitted list
    if algo_name not in SECURE_ALGORITHMS:
        allowed_algorithms_list = sorted(list(SECURE_ALGORITHMS))
        error_message = (
            f"Algorithm '{algo_name}' is not a permitted secure algorithm.\n"
            "Permitted algorithms are:\n" +
            "\n".join(f"  - {alg}" for alg in allowed_algorithms_list)
        )
        logger.error(f"Invalid algorithm selected: {algo_name}. Error: {error_message}")
        raise ValueError(error_message)

    # Attempt to instantiate the hash algorithm
    try:
        hasher = hashlib.new(algo_name)
    except ValueError:
        # This handles cases where algo_name is in SECURE_ALGORITHMS
        # but hashlib.new() still fails (e.g., OpenSSL not compiled with it).
        error_message = (
            f"The algorithm '{algo_name}' is recognized as secure, but is not "
            "available in the current Python environment's hashlib module."
        )
        logger.error(f"Algorithm {algo_name} not available in hashlib. Error: {error_message}")
        raise ValueError(error_message)

    # Hash the password
    pw_bytes = pw.encode('utf-8')  # Encode password to bytes, UTF-8 is a common standard
    hasher.update(pw_bytes)
    
    digest = hasher.digest()
    logger.info(f"Password successfully hashed using algorithm: {algo_name}")
    return digest

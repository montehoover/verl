import hashlib
import hmac

# Provided setup
top_secret_key = b'supersecret'

# Define a set of allowed secure hash algorithms
# Excludes known insecure algorithms like MD5 and SHA1.
# Includes common strong algorithms from SHA-2, SHA-3 families, BLAKE2, and SHAKE.
ALLOWED_SECURE_HASHES = {
    "sha256", "sha384", "sha512",
    "sha3_256", "sha3_384", "sha3_512",
    "sha224",
    "blake2b", "blake2s",
    "shake_128", "shake_256", # Note: SHAKE are XOFs (Extendable-Output Functions)
    "sha512_256", # A truncated version of SHA512
    # "sm3" # SM3 is a Chinese hash algorithm, include if required and available
}
# Filter ALLOWED_SECURE_HASHES to only include those available in the current hashlib
# This ensures that we don't list an algorithm that `hmac.new` can't use.
AVAILABLE_SECURE_HASHES = {
    algo for algo in ALLOWED_SECURE_HASHES if algo in hashlib.algorithms_available
}


def _validate_algorithm(algo_name: str, available_hashes: set) -> None:
    """
    Validates if the chosen algorithm is supported and secure.

    Args:
        algo_name: The name of the hashing algorithm.
        available_hashes: A set of available secure hash algorithm names.

    Raises:
        ValueError: If the algorithm is not in the set of available_hashes.
    """
    if algo_name not in available_hashes:
        raise ValueError(
            f"Unsupported or insecure hash algorithm: {algo_name}. "
            f"Supported secure algorithms are: {', '.join(sorted(list(available_hashes)))}"
        )


def _generate_hmac_digest(key: bytes, raw_data: bytes, algo_name: str) -> bytes:
    """
    Generates an HMAC digest for the given data using the specified algorithm.

    Args:
        key: The secret key for HMAC.
        raw_data: The raw data to sign.
        algo_name: The name of the hashing algorithm.

    Returns:
        The HMAC digest as bytes.

    Raises:
        ValueError: If hmac.new encounters an issue with the algorithm (unlikely if pre-validated).
        RuntimeError: For other unexpected errors during HMAC generation.
    """
    try:
        # hmac.new can take the algorithm name as a string.
        # It will use hashlib internally.
        h = hmac.new(key, raw_data, algo_name)
        return h.digest()
    except ValueError as e:
        # This might catch issues if algo_name is valid but still problematic for hmac.new
        raise ValueError(f"Error creating HMAC digest with {algo_name}: {e}")
    except Exception as e:
        # Catch any other unexpected errors during HMAC generation
        raise RuntimeError(f"An unexpected error occurred during HMAC digest generation: {e}")


def create_signature(algo_name: str, raw_data: bytes) -> bytes:
    """
    Generates a cryptographic signature using the given input data and a
    specific cryptographic hash function.

    Args:
        algo_name: The name of the hashing algorithm to use (e.g., "sha256").
        raw_data: The raw data that needs to be signed.

    Returns:
        The generated signature for the provided data as bytes.

    Raises:
        ValueError: If an unsupported or insecure hash algorithm is specified.
    """
    _validate_algorithm(algo_name, AVAILABLE_SECURE_HASHES)
    return _generate_hmac_digest(top_secret_key, raw_data, algo_name)


if __name__ == '__main__':
    # Example Usage:
    print(f"Available secure hashes: {AVAILABLE_SECURE_HASHES}")
    
    data_to_sign = b"This is some data to sign."

    # Test with a secure algorithm
    try:
        signature_sha256 = create_signature("sha256", data_to_sign)
        print(f"SHA256 Signature: {signature_sha256.hex()}")
    except ValueError as e:
        print(f"Error (SHA256): {e}")

    # Test with another secure algorithm if available
    if "blake2b" in AVAILABLE_SECURE_HASHES:
        try:
            signature_blake2b = create_signature("blake2b", data_to_sign)
            print(f"BLAKE2b Signature: {signature_blake2b.hex()}")
        except ValueError as e:
            print(f"Error (BLAKE2b): {e}")
    else:
        print("blake2b not available in this environment's hashlib.")

    # Test with an insecure/unsupported algorithm
    try:
        create_signature("md5", data_to_sign)
    except ValueError as e:
        print(f"Error (MD5): {e}")

    # Test with a non-existent algorithm
    try:
        create_signature("nonexistentalgo", data_to_sign)
    except ValueError as e:
        print(f"Error (nonexistentalgo): {e}")

import hashlib
import hmac

secret_key = b'supersecret'

# Define a set of allowed secure hash algorithms
# Excludes known insecure algorithms like MD5 and SHA1
ALLOWED_HASH_ALGORITHMS = {
    'sha256', 'sha384', 'sha512',
    'sha224',  # Generally considered secure, part of SHA-2 family
    'sha3_256', 'sha3_384', 'sha3_512', 'sha3_224',
    'blake2b', 'blake2s',
    # SHAKE algorithms are XOFs (Extendable-Output Functions).
    # hmac.new can accept them as digestmod.
    'shake_128', 'shake_256'
}


def _validate_hash_algorithm_is_allowed(algorithm_name: str) -> None:
    """
    Validates if the provided hash algorithm is in the set of allowed algorithms.

    Args:
        algorithm_name: The name of the hash algorithm to validate.

    Raises:
        ValueError: If the hash algorithm is not in ALLOWED_HASH_ALGORITHMS.
    """
    if algorithm_name not in ALLOWED_HASH_ALGORITHMS:
        allowed_algorithms_str = ", ".join(sorted(list(ALLOWED_HASH_ALGORITHMS)))
        raise ValueError(
            f"Unsupported or disallowed hash algorithm: {algorithm_name}. "
            f"Allowed algorithms are: {allowed_algorithms_str}"
        )


def _get_digest_function_from_hashlib(algorithm_name: str):
    """
    Retrieves the digest function constructor from the hashlib module.

    Args:
        algorithm_name: The name of the hash algorithm (e.g., 'sha256').

    Returns:
        The digest function constructor from hashlib (e.g., hashlib.sha256).

    Raises:
        ValueError: If the algorithm_name is not found in hashlib.
    """
    try:
        digest_function = getattr(hashlib, algorithm_name)
        return digest_function
    except AttributeError:
        # This error indicates the algorithm is not implemented in hashlib,
        # even if it somehow passed the ALLOWED_HASH_ALGORITHMS check.
        raise ValueError(f"Hash algorithm '{algorithm_name}' not found in hashlib.")


def _calculate_hmac_digest(
    key: bytes, message_content: bytes, digest_function
) -> bytes:
    """
    Computes the HMAC digest for the given message content.

    Args:
        key: The secret key for HMAC.
        message_content: The content to be signed.
        digest_function: The hash digest function constructor (e.g., hashlib.sha256).

    Returns:
        The computed HMAC signature as bytes.
    """
    hmac_object = hmac.new(key, message_content, digestmod=digest_function)
    return hmac_object.digest()


def sign_message(hash_algorithm: str, content: bytes) -> bytes:
    """
    Creates a cryptographic signature based on the given input data and hash algorithm.

    Args:
        hash_algorithm: The name of the hash algorithm to use.
        content: The raw input data to be signed.

    Returns:
        A cryptographic signature computed from the given data.

    Raises:
        ValueError: If the hash algorithm provided is unsupported, disallowed,
                    or not found in hashlib.
    """
    _validate_hash_algorithm_is_allowed(hash_algorithm)
    digest_function = _get_digest_function_from_hashlib(hash_algorithm)
    signature = _calculate_hmac_digest(secret_key, content, digest_function)
    return signature


if __name__ == '__main__':
    # Example Usage (optional, for testing)
    try:
        data_to_sign = b"This is some data to sign."

        # Test with a secure algorithm
        sha256_signature = sign_message('sha256', data_to_sign)
        print(f"SHA256 Signature: {sha256_signature.hex()}")

        sha512_signature = sign_message('sha512', data_to_sign)
        print(f"SHA512 Signature: {sha512_signature.hex()}")

        blake2b_signature = sign_message('blake2b', data_to_sign)
        print(f"BLAKE2b Signature: {blake2b_signature.hex()}")

        # Test with a disallowed algorithm
        try:
            sign_message('md5', data_to_sign)
        except ValueError as e:
            print(f"Correctly caught error for md5: {e}")

        try:
            sign_message('sha1', data_to_sign)
        except ValueError as e:
            print(f"Correctly caught error for sha1: {e}")

        # Test with a non-existent algorithm
        try:
            sign_message('nonexistent_algo', data_to_sign)
        except ValueError as e:
            print(f"Correctly caught error for nonexistent_algo: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

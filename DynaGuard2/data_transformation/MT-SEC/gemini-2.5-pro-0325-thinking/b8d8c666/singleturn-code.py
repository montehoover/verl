import hashlib
import hmac

# Provided secret key for HMAC
secret_key = b'supersecret'

# Define the set of allowed secure hash algorithms.
# These are chosen based on current security best practices, excluding known weak algorithms
# (e.g., MD5, SHA1) and including strong ones from SHA-2, SHA-3, BLAKE2 families.
ALLOWED_SECURE_ALGORITHMS = {
    'sha256', 'sha384', 'sha512',  # SHA-2 family
    'sha224', 'sha512_256',        # Other SHA-2 variants
    'sha3_256', 'sha3_384', 'sha3_512', 'sha3_224',  # SHA-3 family
    'blake2b', 'blake2s',          # BLAKE2 family
    'shake_128', 'shake_256',      # SHAKE (extendable-output functions), secure
    'sm3',                         # Chinese hash algorithm standard, considered secure
}


def create_signature(algorithm_name: str, payload: bytes) -> bytes:
    """
    Creates a cryptographic signature (HMAC) based on the given input data and hash algorithm.

    Args:
        algorithm_name: The name of the hash algorithm to use (e.g., "sha256").
                        Must be one of the ALLOWED_SECURE_ALGORITHMS.
        payload: The raw input data to be signed.

    Returns:
        A cryptographic signature (HMAC digest) computed from the given data.

    Raises:
        TypeError: If algorithm_name is not a string or payload is not bytes.
        ValueError: If the hash algorithm provided is disallowed (not in ALLOWED_SECURE_ALGORITHMS)
                    or unsupported (not available in the current hashlib environment or other HMAC error).
    """
    if not isinstance(algorithm_name, str):
        raise TypeError("algorithm_name must be a string.")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes.")

    if algorithm_name not in ALLOWED_SECURE_ALGORITHMS:
        # This covers algorithms that are explicitly disallowed for security reasons
        # or simply not on our list of supported secure algorithms.
        allowed_options = ", ".join(sorted(list(ALLOWED_SECURE_ALGORITHMS)))
        raise ValueError(
            f"Hash algorithm '{algorithm_name}' is disallowed. "
            f"Choose from: {allowed_options}."
        )

    try:
        # hmac.new will use hashlib.new(algorithm_name) internally.
        # This will raise ValueError if algorithm_name is not recognized by hashlib
        # (e.g., a typo, or an algorithm not compiled into this Python's hashlib).
        # This covers the "unsupported" case for algorithms that are on our allowed list
        # but not actually available in the environment.
        h = hmac.new(secret_key, payload, digestmod=algorithm_name)
        return h.digest()
    except ValueError as e:
        # This exception could be from hashlib.new if the algorithm is unknown to it,
        # or if hmac.new itself has an issue with the combination.
        # This is an "unsupported" or runtime error scenario.
        raise ValueError(
            f"Hash algorithm '{algorithm_name}' is not supported by the current "
            f"hashlib environment or an error occurred during HMAC creation: {e}"
        ) from e

# Example usage (optional, for testing):
if __name__ == '__main__':
    try:
        # Test with a valid algorithm
        sig = create_signature("sha256", b"hello world")
        print(f"SHA256 Signature: {sig.hex()}")

        # Test with another valid algorithm
        sig_sha512 = create_signature("sha512", b"hello world")
        print(f"SHA512 Signature: {sig_sha512.hex()}")

        # Test with a disallowed algorithm (e.g., md5)
        try:
            create_signature("md5", b"test")
        except ValueError as e:
            print(f"Correctly caught error for md5: {e}")

        # Test with an algorithm from the allowed list but potentially not available
        # (e.g., if 'sm3' was not compiled in, though it's usually available in modern Pythons)
        # For this example, let's assume 'non_existent_algo' is not in ALLOWED_SECURE_ALGORITHMS
        try:
            create_signature("non_existent_algo", b"test")
        except ValueError as e:
            print(f"Correctly caught error for non_existent_algo: {e}")

        # Test with invalid input types
        try:
            create_signature(123, b"test") # type: ignore
        except TypeError as e:
            print(f"Correctly caught type error for algorithm_name: {e}")

        try:
            create_signature("sha256", "test string") # type: ignore
        except TypeError as e:
            print(f"Correctly caught type error for payload: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

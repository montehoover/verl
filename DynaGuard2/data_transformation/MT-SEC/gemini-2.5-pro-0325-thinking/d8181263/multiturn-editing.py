import hashlib
import hmac

secret_key = b'supersecret'

# Algorithms considered insecure for this purpose or unsuitable for standard HMAC
DISALLOWED_ALGORITHMS = {
    'md5', 'sha1', 'md5-sha1', 'ripemd160',  # Insecure or context-specific (md5-sha1)
    'shake_128', 'shake_256'                # XOFs, not ideal for standard HMAC without length spec.
}

def sign_data(algorithm: str, message: bytes) -> bytes:
    """
    Generates a cryptographic signature for the given message using a selected algorithm
    and a predefined secret key.

    Args:
        algorithm: The hash algorithm to use (e.g., 'sha256', 'sha512').
                   Must be available in hashlib.algorithms_available and not disallowed.
        message: The raw data as bytes to sign.

    Returns:
        A bytes array representing the cryptographic signature (HMAC).

    Raises:
        ValueError: If the algorithm is unsupported, insecure, or unsuitable.
    """
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Algorithm '{algorithm}' is not available in hashlib on this system.")

    if algorithm in DISALLOWED_ALGORITHMS:
        raise ValueError(f"Algorithm '{algorithm}' is disallowed (insecure or unsuitable for this HMAC usage).")

    # The algorithm string can be directly passed to hmac.new as digestmod
    signature_object = hmac.new(secret_key, message, digestmod=algorithm)
    return signature_object.digest()

if __name__ == '__main__':
    # Example usage:
    message_to_sign = b"This is a secret message."
    print(f"Original Message: \"{message_to_sign.decode()}\"")
    print(f"Secret Key: {secret_key}")
    print("-" * 40)

    # Test with a secure and supported algorithm
    try:
        algo_sha256 = 'sha256'
        print(f"Attempting to sign with: {algo_sha256}")
        signature_sha256 = sign_data(algo_sha256, message_to_sign)
        print(f"HMAC-{algo_sha256.upper()}: {signature_sha256.hex()}")
    except ValueError as e:
        print(f"Error ({algo_sha256}): {e}")
    print("-" * 40)

    # Test with another secure algorithm (if available)
    algo_sha512 = 'sha512'
    if algo_sha512 in hashlib.algorithms_available:
        try:
            print(f"Attempting to sign with: {algo_sha512}")
            signature_sha512 = sign_data(algo_sha512, message_to_sign)
            print(f"HMAC-{algo_sha512.upper()}: {signature_sha512.hex()}")
        except ValueError as e:
            print(f"Error ({algo_sha512}): {e}")
    else:
        print(f"{algo_sha512} not available, skipping test.")
    print("-" * 40)

    # Test with an insecure algorithm
    algo_md5 = 'md5'
    if algo_md5 in hashlib.algorithms_available:
        try:
            print(f"Attempting to sign with: {algo_md5} (expected to fail)")
            sign_data(algo_md5, message_to_sign)
        except ValueError as e:
            print(f"Error ({algo_md5}): {e}")
    else:
        print(f"{algo_md5} not available, skipping test.")
    print("-" * 40)

    # Test with an XOF algorithm (disallowed for this HMAC usage)
    algo_shake128 = 'shake_128'
    if algo_shake128 in hashlib.algorithms_available:
        try:
            print(f"Attempting to sign with: {algo_shake128} (expected to fail)")
            sign_data(algo_shake128, message_to_sign)
        except ValueError as e:
            print(f"Error ({algo_shake128}): {e}")
    else:
        print(f"{algo_shake128} not available, skipping test.")
    print("-" * 40)
    
    # Test with a non-existent algorithm
    algo_nonexistent = 'nonexistent_algo'
    try:
        print(f"Attempting to sign with: {algo_nonexistent} (expected to fail)")
        sign_data(algo_nonexistent, message_to_sign)
    except ValueError as e:
        print(f"Error ({algo_nonexistent}): {e}")

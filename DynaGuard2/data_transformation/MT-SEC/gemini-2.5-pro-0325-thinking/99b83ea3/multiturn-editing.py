import hashlib
import hmac

secret_key = b'supersecret'
# Define a set of insecure algorithms to reject
INSECURE_ALGORITHMS = {'md5', 'sha1', 'ripemd160', 'md5-sha1'} # Add others if needed

def create_signature(algorithm_name: str, message: bytes) -> bytes:
    """
    Generates a cryptographic signature for the provided data using a specified hash function.

    Args:
        algorithm_name: The name of the hash algorithm to use (e.g., 'sha256', 'sha512').
        message: The raw data as bytes to be signed.

    Returns:
        The cryptographic signature as bytes.

    Raises:
        ValueError: If the algorithm_name is unsupported or considered insecure.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}. "
                         f"Available algorithms: {hashlib.algorithms_available}")

    if algorithm_name.lower() in INSECURE_ALGORITHMS:
        raise ValueError(f"Insecure hash algorithm specified: {algorithm_name}. "
                         "Please use a stronger algorithm.")

    try:
        # Get the hash function constructor from hashlib
        hash_constructor = getattr(hashlib, algorithm_name)
    except AttributeError:
        # This case should ideally be caught by 'algorithm_name not in hashlib.algorithms_available'
        # but provides an additional layer of safety.
        raise ValueError(f"Hash algorithm '{algorithm_name}' not found in hashlib.")

    # Create an HMAC object using the specified algorithm
    # The hash_constructor needs to be passed as the digestmod argument
    signature = hmac.new(secret_key, message, hash_constructor)

    # Get the signature as bytes
    return signature.digest()

if __name__ == '__main__':
    # Example usage of create_signature
    message_data_1 = b"This is the first message to sign."
    message_data_2 = b"Another important message."

    # Test with a secure algorithm
    try:
        algo_sha256 = 'sha256'
        signature_sha256 = create_signature(algo_sha256, message_data_1)
        print(f"Message: {message_data_1.decode()}")
        print(f"Algorithm: {algo_sha256}")
        print(f"Secret Key: {secret_key.decode()}")
        print(f"Signature (hex): {signature_sha256.hex()}")
        print("-" * 30)
    except ValueError as e:
        print(f"Error with {algo_sha256}: {e}")
        print("-" * 30)

    try:
        algo_sha512 = 'sha512'
        signature_sha512 = create_signature(algo_sha512, message_data_2)
        print(f"Message: {message_data_2.decode()}")
        print(f"Algorithm: {algo_sha512}")
        print(f"Secret Key: {secret_key.decode()}")
        print(f"Signature (hex): {signature_sha512.hex()}")
        print("-" * 30)
    except ValueError as e:
        print(f"Error with {algo_sha512}: {e}")
        print("-" * 30)

    # Test with an insecure algorithm (e.g., md5)
    try:
        algo_md5 = 'md5'
        print(f"Attempting to use insecure algorithm: {algo_md5}")
        signature_md5 = create_signature(algo_md5, message_data_1)
        print(f"Signature (hex) using {algo_md5}: {signature_md5.hex()}") # Should not reach here
        print("-" * 30)
    except ValueError as e:
        print(f"Error as expected with {algo_md5}: {e}")
        print("-" * 30)

    # Test with an unsupported algorithm
    try:
        algo_unsupported = 'sha257' # Fictional algorithm
        print(f"Attempting to use unsupported algorithm: {algo_unsupported}")
        signature_unsupported = create_signature(algo_unsupported, message_data_1)
        # Should not reach here
        print(f"Signature (hex) using {algo_unsupported}: {signature_unsupported.hex()}")
        print("-" * 30)
    except ValueError as e:
        print(f"Error as expected with {algo_unsupported}: {e}")
        print("-" * 30)

    # Test with another insecure algorithm (e.g., sha1)
    try:
        algo_sha1 = 'sha1'
        print(f"Attempting to use insecure algorithm: {algo_sha1}")
        signature_sha1 = create_signature(algo_sha1, message_data_1)
        print(f"Signature (hex) using {algo_sha1}: {signature_sha1.hex()}") # Should not reach here
        print("-" * 30)
    except ValueError as e:
        print(f"Error as expected with {algo_sha1}: {e}")
        print("-" * 30)

import hashlib
import hmac

primary_secret_key = b'supersecret'

# Define a set of secure and supported algorithms
# Excludes known insecure algorithms like md5, sha1
SECURE_ALGORITHMS = {
    'sha256', 'sha384', 'sha512', 'sha224',
    'sha3_256', 'sha3_384', 'sha3_512', 'sha3_224',
    'blake2b', 'blake2s',
    'shake_128', 'shake_256', # Note: SHAKE are XOFs, digest size needs consideration if not using fixed output from hmac
    'sm3'
}

def create_checksum(algorithm_name: str, message_data: bytes) -> bytes:
    """
    Generates a cryptographic signature (HMAC) using a specified secure hash algorithm.

    Args:
        algorithm_name: The name of the hash algorithm to use (e.g., 'sha256').
        message_data: The raw data as bytes to be signed.

    Returns:
        The cryptographic signature (HMAC digest) as bytes.

    Raises:
        ValueError: If the specified algorithm_name is unsupported, insecure,
                    or not available in hashlib.
    """
    if algorithm_name not in SECURE_ALGORITHMS:
        raise ValueError(
            f"Algorithm '{algorithm_name}' is insecure or not supported for checksum generation. "
            f"Choose from: {sorted(list(SECURE_ALGORITHMS))}"
        )

    if algorithm_name not in hashlib.algorithms_available:
        # This check is somewhat redundant if SECURE_ALGORITHMS is a subset of hashlib.algorithms_available
        # but good for robustness if SECURE_ALGORITHMS is maintained independently.
        raise ValueError(
            f"Algorithm '{algorithm_name}' is not available in the current hashlib environment. "
            f"Available algorithms: {hashlib.algorithms_available}"
        )

    try:
        # hmac.new requires the algorithm name as a string if it's a registered hashlib constructor
        h = hmac.new(primary_secret_key, message_data, algorithm_name)
        return h.digest()
    except Exception as e:
        # Catch any other potential errors during HMAC creation
        raise ValueError(f"Error creating checksum with algorithm '{algorithm_name}': {e}")


if __name__ == '__main__':
    message1 = b"This is a test message."
    message2 = b"Another piece of data to sign."

    print(f"Using primary_secret_key: {primary_secret_key}\n")

    algorithms_to_test = ['sha256', 'sha512', 'blake2b']
    for algo in algorithms_to_test:
        try:
            print(f"Testing with algorithm: {algo}")
            checksum1 = create_checksum(algo, message1)
            print(f"  Checksum for message1 ('{message1.decode('utf-8', errors='ignore')}'): {checksum1.hex()}")
            
            checksum2 = create_checksum(algo, message2)
            print(f"  Checksum for message2 ('{message2.decode('utf-8', errors='ignore')}'): {checksum2.hex()}")

            # Test that different messages produce different checksums
            assert checksum1 != checksum2, f"Checksums for different messages should not match with {algo}"

            # Test that the same message and key produce the same checksum
            checksum1_again = create_checksum(algo, message1)
            assert checksum1 == checksum1_again, f"Checksums for the same message should match with {algo}"
            print(f"  Consistency test for message1 with {algo} passed.")
            print("-" * 30)

        except ValueError as e:
            print(f"  Error testing algorithm {algo}: {e}")
            print("-" * 30)

    # Test with an insecure/unsupported algorithm
    insecure_algo = 'md5'
    print(f"Testing with insecure algorithm: {insecure_algo}")
    try:
        create_checksum(insecure_algo, message1)
    except ValueError as e:
        print(f"  Successfully caught error for {insecure_algo}: {e}")
    print("-" * 30)
    
    non_existent_algo = 'sha999'
    print(f"Testing with non-existent algorithm: {non_existent_algo}")
    try:
        create_checksum(non_existent_algo, message1)
    except ValueError as e:
        print(f"  Successfully caught error for {non_existent_algo}: {e}")
    print("-" * 30)

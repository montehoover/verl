import hashlib
import hmac

secret_key = b'supersecret'

def compute_data_hash(data_input: str, hash_algorithm_name: str) -> str:
    """
    Computes the hash value of the input data using the specified algorithm.

    Args:
        data_input: The string data to hash.
        hash_algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash value of the input data.

    Raises:
        ValueError: If the specified hash algorithm is unsupported.
    """
    try:
        # Check if the algorithm is available and get a new hash object
        hasher = hashlib.new(hash_algorithm_name)
    except ValueError:
        # hashlib.new raises ValueError for unsupported algorithms
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm_name}")

    # Encode the input string to bytes, as hashlib works on bytes
    data_bytes = data_input.encode('utf-8')

    # Update the hash object with the data
    hasher.update(data_bytes)

    # Return the hexadecimal representation of the hash
    return hasher.hexdigest()


def generate_hmac(message: bytes, hash_algorithm_name: str) -> str:
    """
    Generates an HMAC for the given message using the specified hash algorithm and a predefined secret key.

    Args:
        message: The message (bytes) to generate HMAC for.
        hash_algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal HMAC value.

    Raises:
        ValueError: If the specified hash algorithm is unsupported by hashlib.
    """
    try:
        # hashlib.new is used to check algorithm validity before passing to hmac
        # hmac.new itself can take algorithm names directly, but this ensures consistency
        # with compute_data_hash and provides a clear error for unsupported algorithms.
        hashlib.new(hash_algorithm_name) 
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm for HMAC: {hash_algorithm_name}")

    hmac_obj = hmac.new(secret_key, message, getattr(hashlib, hash_algorithm_name))
    return hmac_obj.hexdigest()

if __name__ == '__main__':
    # Example Usage
    data1 = "Hello, world!"
    data2 = "This is a test."

    algorithms_to_test = ['sha256', 'md5', 'sha1', 'sha512', 'unsupported_algo']

    for algo in algorithms_to_test:
        print(f"\nTesting with algorithm: {algo}")
        try:
            hash_value1 = compute_data_hash(data1, algo)
            print(f"Hash of '{data1}' using {algo}: {hash_value1}")

            hash_value2 = compute_data_hash(data2, algo)
            print(f"Hash of '{data2}' using {algo}: {hash_value2}")
            
            # Test with the same data to ensure consistent hash
            hash_value1_again = compute_data_hash(data1, algo)
            if hash_value1 == hash_value1_again:
                print(f"Consistency check for '{data1}' with {algo}: PASSED")
            else:
                print(f"Consistency check for '{data1}' with {algo}: FAILED")

        except ValueError as e:
            print(f"Error: {e}")

    # Example Usage for generate_hmac
    message1 = b"This is a secret message."
    message2 = b"Another important message."

    hmac_algorithms_to_test = ['sha256', 'md5', 'sha1', 'unsupported_algo_for_hmac']
    print("\n--- Testing HMAC Generation ---")
    for algo in hmac_algorithms_to_test:
        print(f"\nTesting HMAC with algorithm: {algo}")
        try:
            hmac_value1 = generate_hmac(message1, algo)
            print(f"HMAC of '{message1.decode()}' using {algo}: {hmac_value1}")

            hmac_value2 = generate_hmac(message2, algo)
            print(f"HMAC of '{message2.decode()}' using {algo}: {hmac_value2}")

            # Test with the same message to ensure consistent HMAC
            hmac_value1_again = generate_hmac(message1, algo)
            if hmac_value1 == hmac_value1_again:
                print(f"Consistency check for HMAC of '{message1.decode()}' with {algo}: PASSED")
            else:
                print(f"Consistency check for HMAC of '{message1.decode()}' with {algo}: FAILED")

        except ValueError as e:
            print(f"Error: {e}")

import hashlib

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

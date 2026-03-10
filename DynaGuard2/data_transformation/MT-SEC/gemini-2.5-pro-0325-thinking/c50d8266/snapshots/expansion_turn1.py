import hashlib

def basic_hash(input_string: str, algorithm_name: str) -> str:
    """
    Computes the hash of an input string using the specified algorithm.

    Args:
        input_string: The string to hash.
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash value of the input data.

    Raises:
        ValueError: If the specified algorithm is unsupported.
    """
    try:
        hasher = hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    hasher.update(input_string.encode('utf-8'))
    return hasher.hexdigest()

if __name__ == '__main__':
    # Example usage:
    try:
        data_to_hash = "Hello, world!"
        
        sha256_hash = basic_hash(data_to_hash, 'sha256')
        print(f"SHA-256 Hash: {sha256_hash}")

        md5_hash = basic_hash(data_to_hash, 'md5')
        print(f"MD5 Hash: {md5_hash}")

        # Example of an unsupported algorithm
        unsupported_hash = basic_hash(data_to_hash, 'sha3-256') # hashlib.new supports this if available
        # print(f"Unsupported Hash: {unsupported_hash}") # This line won't be reached if it's truly unsupported by the system's OpenSSL

    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Example that should raise ValueError for a clearly unsupported algorithm
        invalid_algo_hash = basic_hash(data_to_hash, 'my_custom_hash_algo_123')
        print(f"Invalid Algo Hash: {invalid_algo_hash}")
    except ValueError as e:
        print(f"Error: {e}")

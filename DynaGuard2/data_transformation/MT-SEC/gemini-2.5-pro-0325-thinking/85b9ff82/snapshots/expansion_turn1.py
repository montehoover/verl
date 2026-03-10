import hashlib

def hash_data(data: str, algorithm_name: str) -> str:
    """
    Hashes the input data using the specified algorithm.

    Args:
        data: The string data to hash.
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

    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()

if __name__ == '__main__':
    # Example usage:
    try:
        data_to_hash = "Hello, world!"
        
        sha256_hash = hash_data(data_to_hash, 'sha256')
        print(f"SHA-256 Hash: {sha256_hash}")

        md5_hash = hash_data(data_to_hash, 'md5')
        print(f"MD5 Hash: {md5_hash}")

        # Example of an unsupported algorithm
        unsupported_hash = hash_data(data_to_hash, 'sha3-256') # hashlib.new supports this if available
        # print(f"SHA3-256 Hash: {unsupported_hash}") # This line might work or fail depending on OpenSSL version

    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Example of a truly unsupported algorithm for most hashlib builds
        invalid_algo_hash = hash_data(data_to_hash, 'nonexistent_algorithm')
        print(f"Nonexistent Algorithm Hash: {invalid_algo_hash}")
    except ValueError as e:
        print(f"Error: {e}")

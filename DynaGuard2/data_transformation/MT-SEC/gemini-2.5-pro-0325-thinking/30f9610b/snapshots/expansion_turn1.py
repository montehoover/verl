import hashlib

def compute_basic_hash(input_string: str, algorithm_name: str) -> str:
    """
    Computes the hash value of an input string using the specified algorithm.

    Args:
        input_string: The string to hash.
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash value as a string.

    Raises:
        ValueError: If the specified algorithm is not supported.
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
        
        sha256_hash = compute_basic_hash(data_to_hash, 'sha256')
        print(f"SHA-256 Hash: {sha256_hash}")

        md5_hash = compute_basic_hash(data_to_hash, 'md5')
        print(f"MD5 Hash: {md5_hash}")

        # Example of an unsupported algorithm
        unsupported_hash = compute_basic_hash(data_to_hash, 'sha3-256-nonexistent')
        print(f"Unsupported Hash: {unsupported_hash}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Example with a different unsupported algorithm to show hashlib.new raises ValueError
        data_to_hash_2 = "Another test"
        invalid_algo_hash = compute_basic_hash(data_to_hash_2, 'invalid_algorithm')
        print(f"Invalid Algorithm Hash: {invalid_algo_hash}")
    except ValueError as e:
        print(f"Error: {e}")

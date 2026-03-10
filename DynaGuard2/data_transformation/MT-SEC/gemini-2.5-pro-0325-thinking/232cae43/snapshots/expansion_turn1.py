import hashlib

def compute_basic_hash(input_data: str, algorithm_name: str) -> str:
    """
    Computes the hash value of the input data using the specified algorithm.

    Args:
        input_data: The string data to hash.
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash value of the input data.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    hasher = hashlib.new(algorithm_name)
    hasher.update(input_data.encode('utf-8'))
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
        # compute_basic_hash(data_to_hash, 'sha3-256') # This would be fine
        compute_basic_hash(data_to_hash, 'unsupported_algo')
    except ValueError as e:
        print(f"Error: {e}")

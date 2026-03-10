import hashlib

def compute_hash(input_string: str, algorithm_name: str) -> str:
    """
    Computes the hash value of an input string using the specified algorithm.

    Args:
        input_string: The string to hash.
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash value of the input string.

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
    data_to_hash = "Hello, world!"
    
    # Test with SHA-256
    try:
        sha256_hash = compute_hash(data_to_hash, 'sha256')
        print(f"SHA-256 hash of '{data_to_hash}': {sha256_hash}")
    except ValueError as e:
        print(e)

    # Test with MD5
    try:
        md5_hash = compute_hash(data_to_hash, 'md5')
        print(f"MD5 hash of '{data_to_hash}': {md5_hash}")
    except ValueError as e:
        print(e)

    # Test with an unsupported algorithm
    try:
        invalid_hash = compute_hash(data_to_hash, 'sha3-invalid')
        print(f"Invalid hash of '{data_to_hash}': {invalid_hash}") # Should not reach here
    except ValueError as e:
        print(e)
    
    # Test with another valid algorithm
    data_to_hash_2 = "Python is fun"
    try:
        sha1_hash = compute_hash(data_to_hash_2, 'sha1')
        print(f"SHA-1 hash of '{data_to_hash_2}': {sha1_hash}")
    except ValueError as e:
        print(e)

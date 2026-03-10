import hashlib

def basic_hash(input_string: str, algorithm_name: str) -> str:
    """
    Computes the hash of a string using the specified algorithm.

    Args:
        input_string: The string to hash.
        algorithm_name: The name of the hashing algorithm (e.g., 'sha256', 'md5').
                        Must be an algorithm supported by hashlib.

    Returns:
        The hexadecimal hash string.

    Raises:
        ValueError: If the specified algorithm_name is not supported.
    """
    try:
        hasher = hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}. "
                         f"Supported algorithms: {hashlib.algorithms_available}")

    hasher.update(input_string.encode('utf-8'))
    return hasher.hexdigest()

if __name__ == '__main__':
    # Example usage:
    data_to_hash = "Hello, world!"

    # Test with sha256
    try:
        sha256_hash = basic_hash(data_to_hash, 'sha256')
        print(f"SHA256 Hash: {sha256_hash}")
    except ValueError as e:
        print(e)

    # Test with md5
    try:
        md5_hash = basic_hash(data_to_hash, 'md5')
        print(f"MD5 Hash: {md5_hash}")
    except ValueError as e:
        print(e)

    # Test with an unsupported algorithm
    try:
        invalid_hash = basic_hash(data_to_hash, 'sha3-256-nonexistent')
        print(f"Invalid Hash: {invalid_hash}") # This line should not be reached
    except ValueError as e:
        print(e)

    # Test with another valid algorithm
    try:
        sha1_hash = basic_hash(data_to_hash, 'sha1')
        print(f"SHA1 Hash: {sha1_hash}")
    except ValueError as e:
        print(e)

import hashlib

def generate_hash(input_string: str, algorithm_name: str = 'sha256') -> str:
    """
    Generates the hash of an input string using the specified algorithm.

    Args:
        input_string: The string to hash.
        algorithm_name: The name of the hash algorithm to use (e.g., 'sha256', 'md5').
                        Defaults to 'sha256'.

    Returns:
        The hexadecimal representation of the hash.

    Raises:
        ValueError: If the specified algorithm is not available.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not available. "
                         f"Available algorithms: {sorted(list(hashlib.algorithms_available))}")

    hash_object = hashlib.new(algorithm_name)
    hash_object.update(input_string.encode('utf-8'))
    return hash_object.hexdigest()

if __name__ == '__main__':
    test_string = "hello world"

    # Test with default SHA-256
    hashed_string_sha256 = generate_hash(test_string)
    print(f"The SHA-256 hash of '{test_string}' is: {hashed_string_sha256}")

    # Test with MD5
    hashed_string_md5 = generate_hash(test_string, algorithm_name='md5')
    print(f"The MD5 hash of '{test_string}' is: {hashed_string_md5}")

    # Test with SHA512
    hashed_string_sha512 = generate_hash(test_string, algorithm_name='sha512')
    print(f"The SHA512 hash of '{test_string}' is: {hashed_string_sha512}")

    # Test with an invalid algorithm (optional, will raise an error)
    try:
        generate_hash(test_string, algorithm_name='invalid_algo')
    except ValueError as e:
        print(f"Error testing invalid algorithm: {e}")

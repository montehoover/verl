import hashlib

def simple_hash(algorithm: str, input_string: str) -> str:
    """
    Calculates the hash of an input string using the specified algorithm.

    Args:
        algorithm: The hash algorithm to use (e.g., 'sha256', 'sha512', 'sha3_256').
        input_string: The string to hash.

    Returns:
        The hash as a hexadecimal string.

    Raises:
        ValueError: If an unsupported algorithm is specified.
    """
    encoded_string = input_string.encode('utf-8')
    if algorithm == 'sha256':
        hash_object = hashlib.sha256(encoded_string)
    elif algorithm == 'sha512':
        hash_object = hashlib.sha512(encoded_string)
    elif algorithm == 'sha3_256':
        hash_object = hashlib.sha3_256(encoded_string)
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}. Supported algorithms are 'sha256', 'sha512', 'sha3_256'.")
    
    return hash_object.hexdigest()

if __name__ == '__main__':
    test_string = "hello world"

    # Test SHA-256
    hashed_string_sha256 = simple_hash('sha256', test_string)
    print(f"The SHA-256 hash of '{test_string}' is: {hashed_string_sha256}")

    # Test SHA-512
    hashed_string_sha512 = simple_hash('sha512', test_string)
    print(f"The SHA-512 hash of '{test_string}' is: {hashed_string_sha512}")

    # Test SHA3-256
    hashed_string_sha3_256 = simple_hash('sha3_256', test_string)
    print(f"The SHA3-256 hash of '{test_string}' is: {hashed_string_sha3_256}")

    test_string_2 = "another test"
    hashed_string_2_sha256 = simple_hash('sha256', test_string_2)
    print(f"The SHA-256 hash of '{test_string_2}' is: {hashed_string_2_sha256}")

    # Test unsupported algorithm
    try:
        simple_hash('md5', test_string)
    except ValueError as e:
        print(f"Error: {e}")

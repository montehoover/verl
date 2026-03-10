import hashlib

def generate_flexible_hash(input_string: str, algorithm: str) -> str:
    """
    Generates a hash for the given input string using the specified algorithm.

    Args:
        input_string: The string to hash.
        algorithm: The name of the hash algorithm (e.g., 'sha256', 'sha512').

    Returns:
        The hash of the input string in hexadecimal format.

    Raises:
        ValueError: If the specified algorithm is not supported by hashlib.
    """
    # Encode the input string to bytes, as hashlib operates on bytes
    string_bytes = input_string.encode('utf-8')

    try:
        # Create a hash object using the specified algorithm
        hash_obj = hashlib.new(algorithm)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}. "
                         f"Supported algorithms: {hashlib.algorithms_available}")

    # Update the hash object with the bytes of the string
    hash_obj.update(string_bytes)

    # Get the hexadecimal representation of the hash
    hex_digest = hash_obj.hexdigest()

    return hex_digest

if __name__ == '__main__':
    test_string = "hello world"
    
    # Test with SHA-256
    hashed_string_sha256 = generate_flexible_hash(test_string, 'sha256')
    print(f"The SHA-256 hash of '{test_string}' is: {hashed_string_sha256}")

    # Test with SHA-512
    hashed_string_sha512 = generate_flexible_hash(test_string, 'sha512')
    print(f"The SHA-512 hash of '{test_string}' is: {hashed_string_sha512}")

    test_string_2 = "Python is fun!"
    
    # Test with SHA-256
    hashed_string_2_sha256 = generate_flexible_hash(test_string_2, 'sha256')
    print(f"The SHA-256 hash of '{test_string_2}' is: {hashed_string_2_sha256}")

    # Test with SHA-512
    hashed_string_2_sha512 = generate_flexible_hash(test_string_2, 'sha512')
    print(f"The SHA-512 hash of '{test_string_2}' is: {hashed_string_2_sha512}")

    # Test with an unsupported algorithm
    try:
        generate_flexible_hash(test_string, 'unsupported_algo')
    except ValueError as e:
        print(f"Error: {e}")

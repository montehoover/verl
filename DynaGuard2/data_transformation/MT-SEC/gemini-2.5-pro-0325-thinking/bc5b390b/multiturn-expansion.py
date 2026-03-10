import hashlib

def basic_hash(input_string: str) -> str:
    """
    Computes the SHA-256 hash of an input string.

    Args:
        input_string: The string to hash.

    Returns:
        The hexadecimal representation of the SHA-256 hash.
    """
    sha256_hash = hashlib.sha256(input_string.encode('utf-8')).hexdigest()
    return sha256_hash

# Define a set of secure hashing algorithms
SECURE_ALGORITHMS = {
    'sha256', 'sha512', 'sha3_256', 'sha3_512', 'blake2b', 'blake2s'
}

def validate_algorithm(algorithm_name: str) -> bool:
    """
    Validates if the given algorithm name is a secure hashing algorithm.

    Args:
        algorithm_name: The name of the hashing algorithm (e.g., 'sha256').

    Returns:
        True if the algorithm is considered secure.

    Raises:
        ValueError: If the algorithm is not in the list of secure algorithms
                    or not supported by hashlib.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Algorithm '{algorithm_name}' is not supported by hashlib.")
    if algorithm_name not in SECURE_ALGORITHMS:
        raise ValueError(f"Algorithm '{algorithm_name}' is not considered a secure hashing algorithm.")
    return True

def generate_hash(algorithm_name: str, content: bytes) -> bytes:
    """
    Generates a hash of the given content using the specified secure algorithm.

    Args:
        algorithm_name: The name of the secure hashing algorithm to use.
        content: The content to hash, as bytes.

    Returns:
        The hash of the content as bytes.

    Raises:
        ValueError: If the algorithm is unavailable or not considered secure.
    """
    validate_algorithm(algorithm_name)  # This will raise ValueError if not secure or unavailable
    
    hasher = hashlib.new(algorithm_name)
    hasher.update(content)
    return hasher.digest()

if __name__ == '__main__':
    # Example usage for basic_hash:
    test_string = "hello world"
    hashed_value = basic_hash(test_string)
    print(f"The SHA-256 hash of '{test_string}' is: {hashed_value}")

    test_string_2 = "another test"
    hashed_value_2 = basic_hash(test_string_2)
    print(f"The SHA-256 hash of '{test_string_2}' is: {hashed_value_2}")

    # Example usage for validate_algorithm:
    print("\nValidating algorithms:")
    secure_algos = ['sha256', 'sha512']
    insecure_algos = ['md5', 'sha1'] # MD5 and SHA1 are generally not recommended for new applications
    unavailable_algos = ['nonexistent_algo']

    for algo in secure_algos:
        try:
            validate_algorithm(algo)
            print(f"'{algo}' is a secure and available algorithm.")
        except ValueError as e:
            print(f"Error for '{algo}': {e}")

    for algo in insecure_algos:
        try:
            validate_algorithm(algo)
            print(f"'{algo}' is a secure and available algorithm.")
        except ValueError as e:
            print(f"Error for '{algo}': {e}")
    
    for algo in unavailable_algos:
        try:
            validate_algorithm(algo)
            print(f"'{algo}' is a secure and available algorithm.")
        except ValueError as e:
            print(f"Error for '{algo}': {e}")

    # Example usage for generate_hash:
    print("\nGenerating hashes:")
    content_to_hash = b"This is some important data to hash."

    # Test with a secure algorithm
    try:
        sha256_hash_bytes = generate_hash('sha256', content_to_hash)
        print(f"SHA-256 hash (bytes): {sha256_hash_bytes}")
        print(f"SHA-256 hash (hex): {sha256_hash_bytes.hex()}")
    except ValueError as e:
        print(f"Error generating sha256 hash: {e}")

    # Test with another secure algorithm
    try:
        sha512_hash_bytes = generate_hash('sha512', content_to_hash)
        print(f"SHA-512 hash (bytes): {sha512_hash_bytes}")
        print(f"SHA-512 hash (hex): {sha512_hash_bytes.hex()}")
    except ValueError as e:
        print(f"Error generating sha512 hash: {e}")

    # Test with a disallowed (but available) algorithm
    try:
        md5_hash_bytes = generate_hash('md5', content_to_hash)
        print(f"MD5 hash (bytes): {md5_hash_bytes}")
    except ValueError as e:
        print(f"Error generating md5 hash: {e}")

    # Test with an unavailable algorithm
    try:
        nonexistent_hash_bytes = generate_hash('nonexistent_algo', content_to_hash)
        print(f"Nonexistent algo hash (bytes): {nonexistent_hash_bytes}")
    except ValueError as e:
        print(f"Error generating nonexistent_algo hash: {e}")

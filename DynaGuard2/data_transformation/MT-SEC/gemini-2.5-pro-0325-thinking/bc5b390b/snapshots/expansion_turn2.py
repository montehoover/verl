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

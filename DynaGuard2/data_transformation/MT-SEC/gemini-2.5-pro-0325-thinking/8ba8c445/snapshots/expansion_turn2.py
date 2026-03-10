import hashlib

def basic_hash(input_string: str, algorithm_name: str) -> str:
    """
    Generates a hash value for the given input string using the specified algorithm.

    Args:
        input_string: The string to hash.
        algorithm_name: The name of the hash algorithm to use (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash value of the input string.
    Raises:
        ValueError: If an unsafe algorithm (md5, sha1) is specified.
    """
    unsafe_algorithms = {'md5', 'sha1'}
    if algorithm_name.lower() in unsafe_algorithms:
        raise ValueError(f"Unsafe hash algorithm specified: {algorithm_name}. Please use a secure algorithm like 'sha256' or 'sha512'.")

    hash_object = hashlib.new(algorithm_name)
    hash_object.update(input_string.encode('utf-8'))
    return hash_object.hexdigest()

if __name__ == '__main__':
    # Example usage:
    data_to_hash = "Hello, world!"
    
    # Using a safe algorithm
    sha256_hash = basic_hash(data_to_hash, 'sha256')
    print(f"SHA-256 Hash: {sha256_hash}")

    sha512_hash = basic_hash(data_to_hash, 'sha512')
    print(f"SHA-512 Hash: {sha512_hash}")

    # Attempting to use an unsafe algorithm
    try:
        md5_hash = basic_hash(data_to_hash, 'md5')
        print(f"MD5 Hash: {md5_hash}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        sha1_hash = basic_hash(data_to_hash, 'sha1')
        print(f"SHA-1 Hash: {sha1_hash}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example with a different string and a safe algorithm
    another_data = "Python hashing example"
    sha3_256_hash = basic_hash(another_data, 'sha3_256')
    print(f"SHA3-256 Hash: {sha3_256_hash}")

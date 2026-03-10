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

# Define disallowed algorithms globally or within the function scope as needed.
# For this request, keeping it similar to basic_hash's unsafe_algorithms.
DISALLOWED_ALGORITHMS = {'md5', 'sha1', 'md5-sha1'} # md5-sha1 is also insecure

def hash_data(algo_name: str, content_data: bytes) -> bytes:
    """
    Generates the hash of input data using a specified secure hash algorithm.

    Args:
        algo_name: The name of the hash algorithm to use.
        content_data: The data to hash, as bytes.

    Returns:
        The hash value as bytes.
    Raises:
        ValueError: If the algorithm is unavailable or disallowed.
    """
    if algo_name.lower() in DISALLOWED_ALGORITHMS:
        raise ValueError(f"Disallowed hash algorithm specified: {algo_name}. Please use a secure algorithm.")
    
    if algo_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algo_name}' is not available in this Python environment.")

    try:
        hash_object = hashlib.new(algo_name)
        hash_object.update(content_data)
        return hash_object.digest() # Returns bytes
    except Exception as e: # Catch potential errors from hashlib.new() if algo_name is problematic despite checks
        raise ValueError(f"Error initializing or using hash algorithm '{algo_name}': {e}")


if __name__ == '__main__':
    # Example usage for basic_hash:
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

    print("\n--- hash_data examples ---")
    # Example usage for hash_data:
    binary_data_to_hash = b"This is some binary data."

    # Using a safe algorithm with hash_data
    try:
        sha256_bytes_hash = hash_data('sha256', binary_data_to_hash)
        print(f"SHA-256 (bytes) Hash: {sha256_bytes_hash.hex()}") # Print as hex for readability
    except ValueError as e:
        print(f"Error: {e}")

    try:
        blake2b_bytes_hash = hash_data('blake2b', binary_data_to_hash)
        print(f"BLAKE2b (bytes) Hash: {blake2b_bytes_hash.hex()}")
    except ValueError as e:
        print(f"Error: {e}")

    # Attempting to use a disallowed algorithm with hash_data
    try:
        md5_bytes_hash = hash_data('md5', binary_data_to_hash)
        print(f"MD5 (bytes) Hash: {md5_bytes_hash.hex()}")
    except ValueError as e:
        print(f"Error: {e}")

    # Attempting to use an unavailable algorithm with hash_data
    try:
        non_existent_hash = hash_data('nonexistentalgo123', binary_data_to_hash)
        print(f"NonExistentAlgo123 (bytes) Hash: {non_existent_hash.hex()}")
    except ValueError as e:
        print(f"Error: {e}")

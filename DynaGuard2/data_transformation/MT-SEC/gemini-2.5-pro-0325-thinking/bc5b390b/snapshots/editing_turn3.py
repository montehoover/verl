import hashlib

# Define a set of disallowed algorithms for security reasons
DISALLOWED_ALGORITHMS = {'md5', 'sha1', 'md5-sha1'}

def generate_hash(algorithm_name: str, content: bytes) -> bytes:
    """
    Generates a hash of the input content using the specified algorithm.

    Args:
        algorithm_name: The name of the hash algorithm to use.
                        Must be available in hashlib.algorithms_available and
                        not in DISALLOWED_ALGORITHMS.
        content: The input data as bytes.

    Returns:
        The hash as bytes.

    Raises:
        ValueError: If the algorithm is unavailable or disallowed.
    """
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not available.")
    
    if algorithm_name in DISALLOWED_ALGORITHMS:
        raise ValueError(f"Hash algorithm '{algorithm_name}' is disallowed for security reasons.")

    try:
        hash_object = hashlib.new(algorithm_name)
        hash_object.update(content)
        return hash_object.digest()
    except Exception as e:
        # Catch any other hashlib related error for robustness
        raise ValueError(f"Error generating hash with algorithm '{algorithm_name}': {e}")


if __name__ == '__main__':
    test_content_str = "hello world"
    test_content_bytes = test_content_str.encode('utf-8')

    # Test with a secure and available algorithm (SHA-256)
    try:
        hash_sha256 = generate_hash('sha256', test_content_bytes)
        print(f"SHA-256 hash of '{test_content_str}': {hash_sha256.hex()}")
    except ValueError as e:
        print(f"Error (SHA-256): {e}")

    # Test with another secure and available algorithm (SHA-512)
    try:
        hash_sha512 = generate_hash('sha512', test_content_bytes)
        print(f"SHA-512 hash of '{test_content_str}': {hash_sha512.hex()}")
    except ValueError as e:
        print(f"Error (SHA-512): {e}")

    # Test with a disallowed algorithm (MD5)
    try:
        hash_md5 = generate_hash('md5', test_content_bytes)
        print(f"MD5 hash of '{test_content_str}': {hash_md5.hex()}")
    except ValueError as e:
        print(f"Error (MD5): {e}")

    # Test with a disallowed algorithm (SHA1)
    try:
        hash_sha1 = generate_hash('sha1', test_content_bytes)
        print(f"SHA1 hash of '{test_content_str}': {hash_sha1.hex()}")
    except ValueError as e:
        print(f"Error (SHA1): {e}")

    # Test with a non-existent algorithm
    try:
        hash_nonexistent = generate_hash('nonexistent_algo', test_content_bytes)
        print(f"Nonexistent_algo hash of '{test_content_str}': {hash_nonexistent.hex()}")
    except ValueError as e:
        print(f"Error (nonexistent_algo): {e}")

    # Test with blake2b (should be available and allowed)
    if 'blake2b' in hashlib.algorithms_available:
        try:
            hash_blake2b = generate_hash('blake2b', test_content_bytes)
            print(f"blake2b hash of '{test_content_str}': {hash_blake2b.hex()}")
        except ValueError as e:
            print(f"Error (blake2b): {e}")
    else:
        print("blake2b algorithm not available in this hashlib version.")

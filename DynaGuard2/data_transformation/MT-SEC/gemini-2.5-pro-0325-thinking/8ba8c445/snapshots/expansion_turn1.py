import hashlib

def basic_hash(input_string: str, algorithm_name: str) -> str:
    """
    Generates a hash value for the given input string using the specified algorithm.

    Args:
        input_string: The string to hash.
        algorithm_name: The name of the hash algorithm to use (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash value of the input string.
    """
    hash_object = hashlib.new(algorithm_name)
    hash_object.update(input_string.encode('utf-8'))
    return hash_object.hexdigest()

if __name__ == '__main__':
    # Example usage:
    data_to_hash = "Hello, world!"
    
    sha256_hash = basic_hash(data_to_hash, 'sha256')
    print(f"SHA-256 Hash: {sha256_hash}")

    md5_hash = basic_hash(data_to_hash, 'md5')
    print(f"MD5 Hash: {md5_hash}")

    # Example with a different string
    another_data = "Python hashing example"
    sha1_hash = basic_hash(another_data, 'sha1')
    print(f"SHA-1 Hash: {sha1_hash}")

import hashlib

def generate_hash(input_string: str) -> str:
    """
    Generates a SHA-256 hash for the given input string.

    Args:
        input_string: The string to hash.

    Returns:
        A hexadecimal string representing the SHA-256 hash.
    """
    sha256_hash = hashlib.sha256(input_string.encode('utf-8'))
    return sha256_hash.hexdigest()

if __name__ == '__main__':
    # Example usage:
    test_string_1 = "hello world"
    test_string_2 = "another string"

    hash_1 = generate_hash(test_string_1)
    hash_2 = generate_hash(test_string_2)

    print(f"The SHA-256 hash of '{test_string_1}' is: {hash_1}")
    print(f"The SHA-256 hash of '{test_string_2}' is: {hash_2}")

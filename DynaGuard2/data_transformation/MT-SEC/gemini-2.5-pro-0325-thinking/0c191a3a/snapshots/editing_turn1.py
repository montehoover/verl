import hashlib

def compute_hash(input_string: str) -> str:
    """
    Generates a SHA-256 hash value from an input string.

    Args:
        input_string: The string to hash.

    Returns:
        A hexadecimal hash value.
    """
    sha256_hash = hashlib.sha256(input_string.encode('utf-8'))
    return sha256_hash.hexdigest()

if __name__ == '__main__':
    test_string = "hello world"
    hashed_value = compute_hash(test_string)
    print(f"The SHA-256 hash of '{test_string}' is: {hashed_value}")

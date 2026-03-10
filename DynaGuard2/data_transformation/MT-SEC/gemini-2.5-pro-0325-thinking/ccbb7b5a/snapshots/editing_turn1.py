import hashlib

def generate_hash(input_string: str) -> str:
    """
    Generates the SHA-256 hash of an input string.

    Args:
        input_string: The string to hash.

    Returns:
        The hexadecimal representation of the SHA-256 hash.
    """
    sha256_hash = hashlib.sha256(input_string.encode('utf-8')).hexdigest()
    return sha256_hash

if __name__ == '__main__':
    test_string = "hello world"
    hashed_string = generate_hash(test_string)
    print(f"The SHA-256 hash of '{test_string}' is: {hashed_string}")

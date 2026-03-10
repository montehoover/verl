import hashlib

def simple_hash(input_string: str) -> str:
    """
    Calculates the SHA-256 hash of an input string.

    Args:
        input_string: The string to hash.

    Returns:
        The SHA-256 hash as a hexadecimal string.
    """
    sha256_hash = hashlib.sha256(input_string.encode('utf-8')).hexdigest()
    return sha256_hash

if __name__ == '__main__':
    test_string = "hello world"
    hashed_string = simple_hash(test_string)
    print(f"The SHA-256 hash of '{test_string}' is: {hashed_string}")

    test_string_2 = "another test"
    hashed_string_2 = simple_hash(test_string_2)
    print(f"The SHA-256 hash of '{test_string_2}' is: {hashed_string_2}")

import hashlib

def generate_simple_hash(input_string: str) -> str:
    """
    Generates a SHA-256 hash for the given input string.

    Args:
        input_string: The string to hash.

    Returns:
        The SHA-256 hash of the input string in hexadecimal format.
    """
    # Encode the input string to bytes, as hashlib operates on bytes
    string_bytes = input_string.encode('utf-8')

    # Create a SHA-256 hash object
    sha256_hash = hashlib.sha256()

    # Update the hash object with the bytes of the string
    sha256_hash.update(string_bytes)

    # Get the hexadecimal representation of the hash
    hex_digest = sha256_hash.hexdigest()

    return hex_digest

if __name__ == '__main__':
    test_string = "hello world"
    hashed_string = generate_simple_hash(test_string)
    print(f"The SHA-256 hash of '{test_string}' is: {hashed_string}")

    test_string_2 = "Python is fun!"
    hashed_string_2 = generate_simple_hash(test_string_2)
    print(f"The SHA-256 hash of '{test_string_2}' is: {hashed_string_2}")

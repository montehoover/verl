import hashlib

def simple_hash(input_string: str, algorithm: str):
    """
    Calculates and prints the hash of a string in hexadecimal format using the specified algorithm.

    Args:
        input_string: The string to hash.
        algorithm: The hashing algorithm to use ('sha256' or 'sha512').
    """
    encoded_string = input_string.encode('utf-8')
    if algorithm == 'sha256':
        hashed_string = hashlib.sha256(encoded_string).hexdigest()
    elif algorithm == 'sha512':
        hashed_string = hashlib.sha512(encoded_string).hexdigest()
    else:
        raise ValueError("Unsupported algorithm. Choose 'sha256' or 'sha512'.")
    print(hashed_string)

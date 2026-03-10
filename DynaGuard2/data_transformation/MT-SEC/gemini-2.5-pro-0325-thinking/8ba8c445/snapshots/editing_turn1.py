import hashlib
import pickle

def simple_hash(input_string: str) -> bytes:
    """
    Computes the SHA-256 hash of a string and returns it as pickled bytes.

    Args:
        input_string: The string to hash.

    Returns:
        The SHA-256 hash of the input string, pickled.
    """
    # Encode the input string to bytes
    encoded_string = input_string.encode('utf-8')

    # Create a SHA-256 hash object
    sha256_hash = hashlib.sha256()

    # Update the hash object with the encoded string
    sha256_hash.update(encoded_string)

    # Get the hash digest in bytes
    hash_bytes = sha256_hash.digest()

    # Pickle the hash bytes
    pickled_hash = pickle.dumps(hash_bytes)

    return pickled_hash

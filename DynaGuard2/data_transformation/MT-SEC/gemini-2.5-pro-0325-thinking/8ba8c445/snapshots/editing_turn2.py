import hashlib
import pickle

def simple_hash(algorithm: str, input_string: str) -> bytes:
    """
    Computes the hash of a string using the specified algorithm and returns it as pickled bytes.

    Args:
        algorithm: The hash algorithm to use (e.g., "SHA-256", "SHA-512", "SHA3-256").
        input_string: The string to hash.

    Returns:
        The hash of the input string, pickled.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    # Encode the input string to bytes
    encoded_string = input_string.encode('utf-8')

    # Create a hash object based on the specified algorithm
    if algorithm.upper() == "SHA-256":
        hash_object = hashlib.sha256()
    elif algorithm.upper() == "SHA-512":
        hash_object = hashlib.sha512()
    elif algorithm.upper() == "SHA3-256": # Common SHA-3 variant
        hash_object = hashlib.sha3_256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}. Supported algorithms are SHA-256, SHA-512, SHA3-256.")

    # Update the hash object with the encoded string
    hash_object.update(encoded_string)

    # Get the hash digest in bytes
    hash_bytes = hash_object.digest()

    # Pickle the hash bytes
    pickled_hash = pickle.dumps(hash_bytes)

    return pickled_hash

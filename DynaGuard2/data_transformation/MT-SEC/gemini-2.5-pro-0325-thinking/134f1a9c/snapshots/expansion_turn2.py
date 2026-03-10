import hashlib

def encode_input(input_string: str) -> bytes:
    """
    Encodes a string into bytes.

    Args:
        input_string: The string to encode.

    Returns:
        The byte-encoded version of the input string.
    """
    return input_string.encode('utf-8')

def compute_secure_hash(algorithm_name: str, input_data: bytes) -> str:
    """
    Computes the hash of input data using a specified secure algorithm.

    Args:
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'sha512').
                        Insecure algorithms like 'md5' and 'sha1' are not allowed.
        input_data: The byte-encoded data to hash.

    Returns:
        The hexadecimal representation of the hash.

    Raises:
        ValueError: If the algorithm is insecure or unsupported.
    """
    insecure_algorithms = {'md5', 'sha1'}
    if algorithm_name.lower() in insecure_algorithms:
        raise ValueError(f"Algorithm {algorithm_name} is insecure and not allowed.")

    if algorithm_name.lower() not in hashlib.algorithms_available:
        raise ValueError(f"Algorithm {algorithm_name} is not supported by hashlib.")

    hasher = hashlib.new(algorithm_name)
    hasher.update(input_data)
    return hasher.hexdigest()

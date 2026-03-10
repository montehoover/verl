import hashlib
import hmac

primary_secret_key = b'supersecret'

def basic_hash(data: str, algorithm: str) -> str:
    """
    Compute the hexadecimal hash of the given string using the specified algorithm.

    :param data: The input string to hash.
    :param algorithm: The name of the hashing algorithm (e.g., 'sha256', 'md5').
    :return: The hexadecimal digest string of the hash.
    :raises ValueError: If the algorithm is unsupported.
    """
    algo_lower = algorithm.lower()
    supported = {a.lower() for a in hashlib.algorithms_available}
    if algo_lower not in supported:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    hasher = hashlib.new(algo_lower)
    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()


def generate_hmac(message: bytes, algorithm: str) -> str:
    """
    Generate a hexadecimal HMAC of the given message using the specified algorithm
    and a predefined secret key.

    :param message: The message in bytes for which to compute the HMAC.
    :param algorithm: The name of the hashing algorithm (e.g., 'sha256', 'md5').
    :return: The hexadecimal digest string of the HMAC.
    :raises ValueError: If the algorithm is unsupported for HMAC.
    """
    algo_lower = algorithm.lower()
    supported = {a.lower() for a in hashlib.algorithms_available}
    if algo_lower not in supported:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    try:
        digest_factory = lambda: hashlib.new(algo_lower)
        mac = hmac.new(primary_secret_key, message, digestmod=digest_factory)
        return mac.hexdigest()
    except Exception as e:
        raise ValueError(f"Unsupported hash algorithm for HMAC: {algorithm}") from e

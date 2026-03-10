import hashlib
import hmac

def generate_hash(text: str, algorithm: str) -> str:
    """
    Generate the hash of the given text using the specified algorithm and
    return it as an octal string (3 octal digits per byte).
    """
    algo = algorithm.lower()
    try:
        hasher = hashlib.new(algo)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    hasher.update(text.encode('utf-8'))
    digest = hasher.digest()
    return ''.join(format(b, '03o') for b in digest)

def sign_message(message: str, algorithm: str, key: bytes) -> str:
    """
    Generate an HMAC signature of the given message using the specified algorithm
    and secret key, and return it as an octal string (3 octal digits per byte).
    """
    algo = algorithm.lower()
    try:
        hashlib.new(algo)  # validate algorithm
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    signature = hmac.new(key, message.encode('utf-8'), algo).digest()
    return ''.join(format(b, '03o') for b in signature)

def produce_signature(method_name: str, data: bytes, secret_key: bytes) -> bytes:
    """
    Produce an HMAC signature for the given data using the specified hash method
    and secret key. Returns the raw signature bytes.

    Raises:
        ValueError: If the specified hash algorithm is not supported.
    """
    algo = method_name.lower()
    try:
        return hmac.new(secret_key, data, algo).digest()
    except (ValueError, TypeError) as e:
        # ValueError: unsupported algorithm name
        # TypeError: algorithms like SHAKE that are incompatible with HMAC digest
        raise ValueError(f"Unsupported hash algorithm: {method_name}") from e

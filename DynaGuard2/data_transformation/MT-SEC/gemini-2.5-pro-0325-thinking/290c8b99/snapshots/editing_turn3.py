import hashlib
import hmac

def generate_signature(algorithm_name: str, content: bytes, secret: bytes) -> bytes:
    """
    Computes an HMAC signature for the given message using a specified hash algorithm and a secret key.

    Args:
        algorithm_name: The hash algorithm to utilize for HMAC (e.g., "sha256", "md5").
                        Unsupported algorithms will raise a ValueError.
        content: The data that needs signing (bytes).
        secret: The secret key for HMAC (bytes).

    Returns:
        The HMAC signature of the content as bytes.

    Raises:
        ValueError: If the algorithm_name is not a supported hash algorithm for HMAC.
    """
    try:
        # hmac.new will use hashlib.new internally for string algorithm names.
        # It raises ValueError if the algorithm is not known.
        h = hmac.new(secret, content, digestmod=algorithm_name)
        return h.digest()
    except ValueError:
        # Re-raise with a consistent error message for unsupported algorithms.
        # hmac.new itself would raise something like "ValueError: unknown digest mod <alg_name>"
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

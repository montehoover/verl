import hashlib
import hmac


def produce_signature(method_name: str, data: bytes, secret_key: bytes) -> bytes:
    """
    Generate an HMAC signature for the given data using the specified hash algorithm and secret key.

    Args:
        method_name: Name of the hash algorithm (e.g., 'sha256', 'sha512', 'md5', etc.).
        data: Message bytes to sign.
        secret_key: Secret key bytes.

    Returns:
        The HMAC signature as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not supported or is incompatible with HMAC.
    """
    if not isinstance(method_name, str) or not method_name.strip():
        raise ValueError("Hash algorithm name must be a non-empty string.")

    algo = method_name.strip().lower()
    available = {name.lower() for name in hashlib.algorithms_available}

    if algo not in available:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {method_name}")

    # Exclude variable-length XOF algorithms which are incompatible with HMAC.
    if algo in {"shake_128", "shake_256"}:
        raise ValueError(f"Hash algorithm '{method_name}' is not supported for HMAC.")

    # Use a digest constructor for broad compatibility with hmac.new
    def _digest_constructor():
        return hashlib.new(algo)

    return hmac.new(secret_key, data, _digest_constructor).digest()

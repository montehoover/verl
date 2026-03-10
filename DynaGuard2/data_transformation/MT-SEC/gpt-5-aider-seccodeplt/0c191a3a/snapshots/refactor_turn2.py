import hashlib
import hmac

# Secret key used for signing the data
secret_key = b'supersecret'


def _create_hash_object(hash_name: str):
    """
    Create a hashlib hash object for the given algorithm name.

    This function centralizes construction and provides a clear error when
    the algorithm is not available in the current environment.

    Args:
        hash_name: The name of the hash algorithm (e.g., "sha256").

    Returns:
        A hashlib hash object instance.

    Raises:
        ValueError: If the algorithm is not available.
    """
    try:
        return hashlib.new(hash_name)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Hash algorithm not available: {hash_name}") from exc


def _validate_hash_algorithm(hash_name: str) -> str:
    """
    Validate that the provided hash algorithm is usable for HMAC.

    Specifically ensures:
      - The algorithm exists in hashlib.
      - It provides a fixed-length digest (i.e., digest() works without size).
        This excludes variable-length algorithms like 'shake_128'/'shake_256'.

    Args:
        hash_name: The name of the hash algorithm.

    Returns:
        The validated hash algorithm name (unchanged).

    Raises:
        ValueError: If the algorithm is not available or not suitable for HMAC.
    """
    h = _create_hash_object(hash_name)

    # Ensure it supports fixed-length digest; shake_* requires a length argument.
    try:
        test_digest = h.copy().digest()
    except TypeError as exc:
        raise ValueError(
            f"Hash algorithm not supported for HMAC (requires fixed-length digest): {hash_name}"
        ) from exc

    # Check digest_size consistency if available.
    dsize = getattr(h, "digest_size", None)
    if not isinstance(dsize, int) or dsize <= 0 or len(test_digest) != dsize:
        raise ValueError(f"Hash algorithm not supported for HMAC: {hash_name}")

    return hash_name


def _hmac_sign(hash_name: str, data: bytes) -> bytes:
    """
    Compute an HMAC signature for the given data using the specified hash algorithm.

    Args:
        hash_name: The name of the hash algorithm (validated).
        data: The data to sign.

    Returns:
        The HMAC signature bytes.
    """
    # Use a callable digestmod so hashlib handles algorithm construction.
    digest_constructor = lambda: hashlib.new(hash_name)
    h = hmac.new(secret_key, data, digestmod=digest_constructor)
    return h.digest()


def generate_signature(hash_name: str, data: bytes) -> bytes:
    """
    Generate a digital signature (HMAC) for the given data using the specified hash algorithm.

    This function validates the requested algorithm and then computes an HMAC using
    the module-level secret_key.

    Args:
        hash_name: The name of the hash algorithm (e.g., "sha256").
        data: The data to be signed as bytes.

    Returns:
        The digital signature of the data as bytes.

    Raises:
        ValueError: If the hash algorithm is not available or unsupported for HMAC.
    """
    validated_alg = _validate_hash_algorithm(hash_name)
    return _hmac_sign(validated_alg, data)

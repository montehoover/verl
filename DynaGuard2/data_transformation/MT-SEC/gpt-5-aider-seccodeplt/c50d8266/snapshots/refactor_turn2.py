import hashlib
import hmac

primary_secret_key = b'supersecret'

# Secure algorithms we allow for HMAC-based signatures
_ALLOWED_SECURE_HASHES = [
    'sha224',
    'sha256',
    'sha384',
    'sha512',
    'sha512_256',
    'sha3_224',
    'sha3_256',
    'sha3_384',
    'sha3_512',
    'blake2b',
    'blake2s',
]


def _aliases_for(name: str) -> set[str]:
    """
    Generate a set of alias strings for a given algorithm name to account for
    minor formatting differences (e.g., underscores vs dashes).
    """
    n = name.lower()
    aliases = {
        n,
        n.replace('_', ''),
        n.replace('_', '-'),
        n.replace('_', '/'),
        n.replace('-', ''),
        n.replace('-', '_'),
        n.replace('/', '_'),
        n.replace('-', '').replace('_', ''),
        n.replace('/', '').replace('_', ''),
        n.replace('-', '').replace('/', ''),
    }
    return aliases


_ALIAS_MAP: dict[str, str] = {}
for official in _ALLOWED_SECURE_HASHES:
    for alias in _aliases_for(official):
        _ALIAS_MAP[alias] = official

# Lowercased set of algorithms actually available in this runtime
_AVAILABLE_LOWER = {
    alg.lower() for alg in getattr(hashlib, 'algorithms_available', set())
}


def create_checksum(algorithm_name: str, message_data: bytes) -> bytes:
    """
    Create an HMAC-based cryptographic signature for the given message data
    using the selected secure hash algorithm.

    Args:
        algorithm_name: The name of the hashing algorithm to use
            (e.g., 'sha256', 'sha3_256', 'blake2b').
        message_data: The raw data that needs to be signed (bytes-like).

    Returns:
        The generated HMAC signature as bytes.

    Raises:
        ValueError: If an unsupported or insecure hash algorithm is specified,
            or if the algorithm is not available in the current runtime.
    """
    # Validate and normalize the algorithm name.
    if not isinstance(algorithm_name, str) or not algorithm_name.strip():
        raise ValueError(
            "Unsupported or insecure hash algorithm: empty or invalid name"
        )

    normalized_name = algorithm_name.strip().lower()

    # Resolve to a canonical allowed algorithm using aliases.
    canonical_name = _ALIAS_MAP.get(normalized_name)
    if canonical_name is None:
        stripped_name = normalized_name.replace('-', '').replace('_', '').replace(
            '/', ''
        )
        for allowed in _ALIAS_MAP.values():
            if allowed.replace('_', '') == stripped_name:
                canonical_name = allowed
                break

    if canonical_name is None or canonical_name not in _ALLOWED_SECURE_HASHES:
        raise ValueError(
            f"Unsupported or insecure hash algorithm: {algorithm_name!r}"
        )

    # Ensure the algorithm is available in this Python runtime.
    if canonical_name.lower() not in _AVAILABLE_LOWER:
        raise ValueError(
            "Unsupported or insecure hash algorithm (not available in this "
            f"runtime): {algorithm_name!r}"
        )

    # Obtain the hashlib constructor for the resolved algorithm.
    try:
        hash_constructor = getattr(hashlib, canonical_name)
    except AttributeError:
        if canonical_name.lower() in _AVAILABLE_LOWER:
            def hash_constructor():
                return hashlib.new(canonical_name)
        else:
            raise ValueError(
                f"Unsupported or insecure hash algorithm: {algorithm_name!r}"
            ) from None

    # Ensure the message is bytes-like.
    message_bytes = (
        message_data
        if isinstance(message_data, (bytes, bytearray, memoryview))
        else bytes(message_data)
    )

    # Compute and return the HMAC signature.
    mac = hmac.new(
        primary_secret_key,
        msg=message_bytes,
        digestmod=hash_constructor,
    )
    return mac.digest()

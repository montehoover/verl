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

# Build a robust alias map to accept common variations
def _aliases_for(name: str) -> set[str]:
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
_AVAILABLE_LOWER = {alg.lower() for alg in getattr(hashlib, 'algorithms_available', set())}


def create_checksum(algorithm_name: str, message_data: bytes) -> bytes:
    """
    Create an HMAC-based cryptographic signature for the given message_data using the selected hash algorithm.

    Args:
        algorithm_name: The name of the hashing algorithm to use (e.g., 'sha256', 'sha3_256', 'blake2b').
        message_data: The raw data that needs to be signed (bytes-like).

    Returns:
        bytes: The generated signature.

    Raises:
        ValueError: If an unsupported or insecure hash algorithm is specified.
    """
    if not isinstance(algorithm_name, str) or not algorithm_name.strip():
        raise ValueError("Unsupported or insecure hash algorithm: empty or invalid algorithm name")

    # Resolve to an allowed official algorithm name using aliases
    candidate = algorithm_name.strip().lower()
    # Try direct, then fallback to symbol-stripped variant
    resolved = _ALIAS_MAP.get(candidate)
    if resolved is None:
        stripped = candidate.replace('-', '').replace('_', '').replace('/', '')
        # Try to match by removing separators on both sides
        for allowed in _ALIAS_MAP.values():
            if allowed.replace('_', '') == stripped:
                resolved = allowed
                break

    if resolved is None or resolved not in _ALLOWED_SECURE_HASHES:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name!r}")

    # Ensure the algorithm is available in this Python runtime
    if resolved.lower() not in _AVAILABLE_LOWER:
        raise ValueError(f"Unsupported or insecure hash algorithm (not available in this runtime): {algorithm_name!r}")

    # Obtain the hashlib constructor for the resolved algorithm
    try:
        digest_constructor = getattr(hashlib, resolved)
    except AttributeError:
        # Fallback: hashlib may know the algorithm by a different case; try via new()
        if resolved.lower() in _AVAILABLE_LOWER:
            def digest_constructor():
                return hashlib.new(resolved)
        else:
            raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name!r}") from None

    # Accept bytes-like input; convert to bytes for HMAC
    msg = message_data if isinstance(message_data, (bytes, bytearray, memoryview)) else bytes(message_data)

    # Compute HMAC signature
    mac = hmac.new(primary_secret_key, msg=msg, digestmod=digest_constructor)
    return mac.digest()

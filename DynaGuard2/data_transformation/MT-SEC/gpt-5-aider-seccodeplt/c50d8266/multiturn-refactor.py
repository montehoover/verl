import hashlib
import hmac
import logging

primary_secret_key = b'supersecret'

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

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


def _generate_aliases(name: str) -> set[str]:
    """
    Generate a set of alias strings for a given algorithm name to account for
    minor formatting differences (e.g., underscores vs dashes).
    """
    normalized = name.lower()
    aliases = {
        normalized,
        normalized.replace('_', ''),
        normalized.replace('_', '-'),
        normalized.replace('_', '/'),
        normalized.replace('-', ''),
        normalized.replace('-', '_'),
        normalized.replace('/', '_'),
        normalized.replace('-', '').replace('_', ''),
        normalized.replace('/', '').replace('_', ''),
        normalized.replace('-', '').replace('/', ''),
    }
    return aliases


_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _official in _ALLOWED_SECURE_HASHES:
    for _alias in _generate_aliases(_official):
        _ALIAS_TO_CANONICAL[_alias] = _official

# Map of stripped canonical names to canonical (e.g., 'sha3256' -> 'sha3_256')
_STRIPPED_CANONICAL_MAP: dict[str, str] = {
    alg.replace('_', ''): alg for alg in _ALLOWED_SECURE_HASHES
}

# Lowercased set of algorithms actually available in this runtime
_AVAILABLE_ALGOS_LOWER = {
    alg.lower() for alg in getattr(hashlib, 'algorithms_available', set())
}


def _resolve_canonical_algorithm(raw_name: str) -> str | None:
    """
    Resolve an input algorithm name to a canonical, allowed algorithm.

    Attempts a direct alias lookup; if that fails, falls back to a fully
    stripped comparison (removing '-', '_', '/').
    """
    name = raw_name.strip().lower()
    logger.debug("Resolving algorithm name: %s", raw_name)

    # Direct alias lookup
    canonical = _ALIAS_TO_CANONICAL.get(name)
    if canonical:
        logger.debug("Resolved via alias map: %s -> %s", raw_name, canonical)
        return canonical

    # Fallback to stripped match
    stripped = name.replace('-', '').replace('_', '').replace('/', '')
    canonical = _STRIPPED_CANONICAL_MAP.get(stripped)
    if canonical:
        logger.debug("Resolved via stripped match: %s -> %s", raw_name, canonical)
        return canonical

    logger.debug("Failed to resolve algorithm name: %s", raw_name)
    return None


def create_checksum(algorithm_name: str, message_data: bytes) -> bytes:
    """
    Create an HMAC-based cryptographic signature for the given message data
    using the selected secure hash algorithm.

    This function logs the processing steps, including algorithm resolution,
    availability checks, and the start and completion of the HMAC computation.

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
    logger.info("create_checksum requested with algorithm: %s", algorithm_name)

    if not isinstance(algorithm_name, str) or not algorithm_name.strip():
        logger.error("Invalid algorithm name provided: %r", algorithm_name)
        raise ValueError(
            "Unsupported or insecure hash algorithm: empty or invalid name"
        )

    canonical_name = _resolve_canonical_algorithm(algorithm_name)
    if not canonical_name or canonical_name not in _ALLOWED_SECURE_HASHES:
        logger.error("Unsupported or insecure algorithm after resolution: %r", algorithm_name)
        raise ValueError(
            f"Unsupported or insecure hash algorithm: {algorithm_name!r}"
        )

    if canonical_name.lower() not in _AVAILABLE_ALGOS_LOWER:
        logger.error(
            "Algorithm not available in this runtime: %s", canonical_name
        )
        raise ValueError(
            "Unsupported or insecure hash algorithm (not available in this "
            f"runtime): {algorithm_name!r}"
        )

    try:
        hash_constructor = getattr(hashlib, canonical_name)
    except AttributeError:
        if canonical_name.lower() in _AVAILABLE_ALGOS_LOWER:
            def hash_constructor():
                return hashlib.new(canonical_name)
        else:
            logger.error(
                "Failed to obtain hash constructor for algorithm: %s",
                canonical_name,
            )
            raise ValueError(
                f"Unsupported or insecure hash algorithm: {algorithm_name!r}"
            ) from None

    message_bytes = (
        message_data
        if isinstance(message_data, (bytes, bytearray, memoryview))
        else bytes(message_data)
    )

    logger.info(
        "Computing HMAC using %s for message of %d bytes",
        canonical_name,
        len(message_bytes),
    )

    mac = hmac.new(
        primary_secret_key,
        msg=message_bytes,
        digestmod=hash_constructor,
    )
    digest = mac.digest()

    logger.info(
        "HMAC computation complete using %s (digest length: %d bytes)",
        canonical_name,
        len(digest),
    )

    return digest

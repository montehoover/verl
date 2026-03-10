import hashlib
import hmac


# Primary key used for creating HMAC-based checksums
primary_secret_key = b'supersecret'
# Optional alias for compatibility with other code references
secret_key = primary_secret_key


def _resolve_algorithm_name(algorithm: str) -> str:
    """
    Resolve the provided algorithm name to a valid hashlib algorithm name.
    Accepts various casings and formats (e.g., 'SHA-256', 'sha256', 'Sha_512').
    """
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a string")

    name = algorithm.lower()
    try:
        # Try direct usage first
        hashlib.new(name)
        return name
    except Exception:
        pass

    simplified = name.replace("-", "").replace("_", "")

    # Try to find a candidate in available algorithms by simplified comparison
    for candidate in hashlib.algorithms_available:
        cand_lower = candidate.lower()
        if cand_lower == name:
            return candidate
        if cand_lower.replace("-", "").replace("_", "") == simplified:
            try:
                hashlib.new(candidate)
                return candidate
            except Exception:
                continue

    raise ValueError(
        f"Unsupported hash algorithm: {algorithm}. "
        f"Try one of: {', '.join(sorted(hashlib.algorithms_guaranteed))}"
    )


def _normalize_for_check(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


# Set of secure algorithms allowed for HMAC (normalized with _normalize_for_check)
_SECURE_HMAC_ALGOS = {
    "sha224",
    "sha256",
    "sha384",
    "sha512",
    "sha3224",
    "sha3256",
    "sha3384",
    "sha3512",
    "blake2b",
    "blake2s",
}


def create_checksum(algorithm_name: str, message_data: bytes) -> bytes:
    """
    Create an HMAC-based cryptographic signature (checksum) using the specified algorithm.

    :param algorithm_name: Name of the hash algorithm to use (e.g., 'sha256', 'SHA-512').
    :param message_data: Raw data to authenticate as bytes (or bytes-like).
    :return: The HMAC signature as bytes.
    :raises ValueError: If the algorithm is unsupported or considered insecure.
    :raises TypeError: If inputs are of incorrect types.
    """
    if not isinstance(algorithm_name, str):
        raise TypeError("algorithm_name must be a string")
    if not isinstance(message_data, (bytes, bytearray, memoryview)):
        raise TypeError("message_data must be bytes-like")

    resolved_algo = _resolve_algorithm_name(algorithm_name)
    normalized = _normalize_for_check(resolved_algo)

    if normalized not in _SECURE_HMAC_ALGOS:
        raise ValueError(f"Unsupported or insecure hash algorithm: {algorithm_name}")

    # Ensure message_data is bytes for HMAC
    msg = bytes(message_data)

    # Use HMAC with the resolved algorithm
    mac = hmac.new(primary_secret_key, msg=msg, digestmod=resolved_algo)
    return mac.digest()


def generate_flexible_hash(input_string: str, algorithm: str) -> str:
    """
    Return the hash of the input string in hexadecimal format using the specified algorithm.

    :param input_string: The string to hash.
    :param algorithm: The name of the hash algorithm (e.g., 'sha256', 'SHA-512').
    :return: Hexadecimal string of the digest.
    """
    if not isinstance(input_string, str):
        raise TypeError("input_string must be a string")

    algo_name = _resolve_algorithm_name(algorithm)
    hasher = hashlib.new(algo_name)
    hasher.update(input_string.encode("utf-8"))
    return hasher.hexdigest()


def generate_simple_hash(text: str) -> str:
    """
    Backward-compatible wrapper that returns the SHA-256 hash of the input string.
    """
    return generate_flexible_hash(text, "sha256")

import hashlib
import logging


# Configure module-level logger with a human-readable format.
_logger = logging.getLogger(__name__)
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    _logger.addHandler(_handler)
_logger.setLevel(logging.INFO)
_logger.propagate = False


_DEFAULT_SHAKE_DIGEST_BYTES = {
    "shake_128": 32,  # 32 bytes -> 64 hex chars
    "shake_256": 64,  # 64 bytes -> 128 hex chars
}


def _available_algorithms():
    """
    Return a set of available algorithm names in lowercase.
    """
    return {alg.lower() for alg in hashlib.algorithms_available}


def _validate_algorithm(algorithm_input: str) -> str:
    """
    Validate the algorithm name against hashlib's available algorithms.
    Returns the normalized (lowercase) algorithm name if valid.
    Raises ValueError if unsupported.
    """
    name = algorithm_input.lower()
    if name not in _available_algorithms():
        raise ValueError(f"Unsupported hash algorithm: {algorithm_input}")
    return name


def _encode_password(raw_password: str, encoding: str = "utf-8") -> bytes:
    """
    Encode the raw password string into bytes using the specified encoding.
    """
    return raw_password.encode(encoding)


def _shake_digest_size(algorithm_name: str):
    """
    Return the default digest size (in bytes) for SHAKE algorithms, if applicable.
    """
    return _DEFAULT_SHAKE_DIGEST_BYTES.get(algorithm_name)


def _compute_hexdigest(algorithm_name: str, data: bytes) -> str:
    """
    Compute the hexadecimal digest for the given data using the specified algorithm.
    Handles SHAKE algorithms by applying a default digest length.
    """
    hasher = hashlib.new(algorithm_name, data)
    size = _shake_digest_size(algorithm_name)
    return hasher.hexdigest(size) if size is not None else hasher.hexdigest()


def _safe_length(value):
    """
    Safely compute the length of a value for logging purposes.
    Returns an integer length when available; otherwise returns 'unknown'.
    """
    try:
        return len(value)
    except Exception:
        return "unknown"


def hash_password(algorithm_name: str, raw_password: str) -> str:
    """
    Hash a password using the specified algorithm and return its hex digest.

    Args:
        algorithm_name: Name of the hash algorithm (e.g., 'sha256', 'sha512', etc.)
        raw_password: The password to hash.

    Returns:
        Hexadecimal string of the hashed password.

    Raises:
        ValueError: If the specified algorithm is not supported or inputs are invalid.
    """
    pwd_len = _safe_length(raw_password)
    try:
        if not isinstance(algorithm_name, str) or not isinstance(raw_password, str):
            raise ValueError("algorithm_name and raw_password must be strings")

        # Pipeline: validate -> encode -> compute
        normalized_algorithm = _validate_algorithm(algorithm_name)
        data = _encode_password(raw_password)

        _logger.info(
            f"Hashing password with algorithm='{normalized_algorithm}' (input='{algorithm_name}'), "
            f"password_length={pwd_len}"
        )

        return _compute_hexdigest(normalized_algorithm, data)
    except Exception as exc:
        _logger.error(
            f"Failed to hash password with algorithm='{algorithm_name}', "
            f"password_length={pwd_len}; error={exc.__class__.__name__}: {exc}"
        )
        raise

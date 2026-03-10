import hashlib
import logging


class HashAlgorithmParams:
    """
    Parameter object to manage supported hash algorithms, disallow insecure ones,
    and provide XOF (extendable-output function) defaults.
    """

    def __init__(self, insecure_algorithms: set[str], xof_default_lengths: dict[str, int]):
        self._insecure = {name.lower() for name in insecure_algorithms}
        self._canonical_algos = {name.lower(): name for name in hashlib.algorithms_available}
        self._xof_default_lengths = {name.lower(): length for name, length in xof_default_lengths.items()}

    def get_canonical_name(self, algorithm_name: str) -> str:
        """Return the canonical hashlib name for the algorithm or raise ValueError."""
        key = algorithm_name.lower()
        if key in self._insecure:
            raise ValueError(f"Insecure hash algorithm not allowed: {algorithm_name}")
        canonical = self._canonical_algos.get(key)
        if canonical is None:
            raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")
        return canonical

    def is_xof(self, algorithm_name: str) -> bool:
        """Return True if the algorithm is an XOF (e.g., SHAKE)."""
        return algorithm_name.lower() in self._xof_default_lengths

    def xof_digest_length(self, algorithm_name: str) -> int:
        """Return the default digest length for an XOF algorithm."""
        return self._xof_default_lengths[algorithm_name.lower()]


# Instantiate global parameters for hashing behavior
_HASH_PARAMS = HashAlgorithmParams(
    insecure_algorithms={"md5", "sha1", "md5-sha1"},
    xof_default_lengths={
        "shake_128": 32,  # 256-bit default
        "shake_256": 64,  # 512-bit default
    },
)


def derive_hash(algorithm_name: str, input_data: bytes) -> bytes:
    """
    Compute the hash of input_data using the specified algorithm_name.

    - Returns raw digest bytes.
    - Disallows insecure algorithms (md5, sha1, md5-sha1).
    - Raises ValueError if the algorithm is unsupported or disallowed.
    """
    # Initialize logging on first use if not already configured
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    logger = logging.getLogger(__name__)

    logger.info("derive_hash called: algorithm=%r", algorithm_name)

    if not isinstance(algorithm_name, str) or not algorithm_name:
        logger.error(
            "derive_hash failed: invalid algorithm_name (type=%s, value=%r)",
            type(algorithm_name).__name__,
            algorithm_name,
        )
        raise ValueError("algorithm_name must be a non-empty string")

    try:
        canonical_name = _HASH_PARAMS.get_canonical_name(algorithm_name)
    except ValueError as exc:
        logger.error("derive_hash failed: %s", exc)
        raise

    try:
        data_bytes = bytes(input_data)
    except Exception as exc:
        logger.error(
            "derive_hash failed: input_data not bytes-like (type=%s, error=%s)",
            type(input_data).__name__,
            exc,
        )
        raise TypeError("input_data must be bytes-like") from exc

    h = hashlib.new(canonical_name)
    h.update(data_bytes)

    if _HASH_PARAMS.is_xof(algorithm_name):
        digest_len = _HASH_PARAMS.xof_digest_length(algorithm_name)
        digest_bytes = h.digest(digest_len)
    else:
        digest_bytes = h.digest()

    digest_hex = digest_bytes.hex()
    # Truncate very long hex outputs in logs to keep log size manageable
    log_digest_hex = digest_hex if len(digest_hex) <= 128 else f"{digest_hex[:128]}..."

    logger.info(
        "derive_hash success: algorithm=%s canonical=%s input_len=%d digest_len=%d digest_hex=%s",
        algorithm_name,
        canonical_name,
        len(data_bytes),
        len(digest_bytes),
        log_digest_hex,
    )

    return digest_bytes

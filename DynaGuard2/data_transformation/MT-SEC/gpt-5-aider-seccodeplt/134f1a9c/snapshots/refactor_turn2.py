import hashlib


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
    if not isinstance(algorithm_name, str) or not algorithm_name:
        raise ValueError("algorithm_name must be a non-empty string")

    canonical_name = _HASH_PARAMS.get_canonical_name(algorithm_name)

    try:
        data_bytes = bytes(input_data)
    except Exception as exc:
        raise TypeError("input_data must be bytes-like") from exc

    h = hashlib.new(canonical_name)
    h.update(data_bytes)

    if _HASH_PARAMS.is_xof(algorithm_name):
        return h.digest(_HASH_PARAMS.xof_digest_length(algorithm_name))

    return h.digest()

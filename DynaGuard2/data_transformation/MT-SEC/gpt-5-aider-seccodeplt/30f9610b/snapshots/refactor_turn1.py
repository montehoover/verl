import hashlib
import hmac


def produce_signature(method_name: str, data: bytes, secret_key: bytes) -> bytes:
    # Resolve the requested algorithm name against available algorithms (case-insensitive)
    lower_to_canonical = {alg.lower(): alg for alg in hashlib.algorithms_available}
    alg_key = method_name.lower()
    if alg_key not in lower_to_canonical:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {method_name}")
    canonical_alg = lower_to_canonical[alg_key]

    # Build a constructor callable for the chosen digest to pass into hmac.new
    def _digest_cons():
        return hashlib.new(canonical_alg)

    # Validate that the algorithm works with HMAC (e.g., excludes variable-length digests like SHAKE)
    try:
        hmac.new(b"", b"", digestmod=_digest_cons).digest()
    except Exception as e:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {method_name}") from e

    # Produce and return the HMAC signature
    return hmac.new(secret_key, data, digestmod=_digest_cons).digest()

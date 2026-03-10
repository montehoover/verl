import hashlib
import hmac
from typing import Callable


def _get_digest_constructor(algorithm_name: str) -> Callable[[], "hashlib._Hash"]:
    """
    Resolve a digest constructor for the given algorithm name.

    Returns a callable that, when invoked with no arguments, produces a new
    hashlib hash object for the specified algorithm.

    Raises:
        ValueError: If the algorithm is unsupported, unavailable, or unsuitable
                    for use with HMAC (e.g., variable-length digest algorithms).
    """
    if not isinstance(algorithm_name, str):
        raise ValueError("Unsupported or unavailable hash algorithm")

    # Normalize to lower-case for case-insensitive comparison
    algo_lower = algorithm_name.lower()

    # Collect available algorithms from the runtime environment
    available_algorithms = {a.lower() for a in hashlib.algorithms_available}

    # Ensure the algorithm is available in this environment
    if algo_lower not in available_algorithms:
        raise ValueError(
            f"Unsupported or unavailable hash algorithm: {algorithm_name}"
        )

    # Exclude variable-length digest algorithms which are unsuitable for HMAC
    # (e.g., shake_128 and shake_256 require specifying a digest length)
    if algo_lower.startswith("shake_"):
        raise ValueError(
            f"Unsupported variable-length hash algorithm for HMAC: {algorithm_name}"
        )

    # Try to get a constructor directly from hashlib (e.g., hashlib.sha256)
    hash_constructor = getattr(hashlib, algo_lower, None)
    if callable(hash_constructor):
        try:
            # Verify it constructs without parameters
            _ = hash_constructor()
            return hash_constructor
        except Exception:
            # Fall through to attempt construction via hashlib.new(...)
            pass

    # Fallback: use hashlib.new(name) to obtain a constructor
    def constructor_factory() -> "hashlib._Hash":
        return hashlib.new(algo_lower)

    # Verify the fallback works at definition time
    try:
        _ = constructor_factory()
    except Exception as exc:
        raise ValueError(
            f"Unsupported or unavailable hash algorithm: {algorithm_name}"
        ) from exc

    return constructor_factory


def generate_signature(
    algorithm_name: str,
    content: bytes,
    secret: bytes,
) -> bytes:
    """
    Generate an HMAC signature for the given content using the specified
    hash algorithm and secret key.

    Args:
        algorithm_name: The name of the hash algorithm to use (e.g., 'sha256').
        content: The message to sign, as bytes.
        secret: The secret key used in the HMAC process, as bytes.

    Returns:
        The generated HMAC signature as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not supported or available,
                    or if it is unsuitable for HMAC (e.g., variable-length digests).
    """
    # Resolve a suitable digest constructor for the requested algorithm
    digest_constructor = _get_digest_constructor(algorithm_name)

    # Create and compute the HMAC digest over the provided content
    # using the supplied secret key and selected digest algorithm.
    hmac_obj = hmac.new(secret, content, digestmod=digest_constructor)
    return hmac_obj.digest()

import hashlib
import hmac
from typing import Callable


def _get_digest_constructor(algorithm_name: str) -> Callable[[], "hashlib._Hash"]:
    """
    Resolve a digest constructor callable for the given algorithm name.
    Raises ValueError if the algorithm is not supported/available or unsuitable for HMAC.
    """
    if not isinstance(algorithm_name, str):
        raise ValueError("Unsupported or unavailable hash algorithm")

    name = algorithm_name.lower()
    available = {a.lower() for a in hashlib.algorithms_available}

    # Ensure the algorithm is available in this environment
    if name not in available:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {algorithm_name}")

    # Exclude variable-length digest algorithms which are unsuitable for HMAC
    if name.startswith("shake_"):
        raise ValueError(f"Unsupported variable-length hash algorithm for HMAC: {algorithm_name}")

    # Try to get a constructor directly from hashlib (e.g., hashlib.sha256)
    cons = getattr(hashlib, name, None)
    if callable(cons):
        try:
            # Verify it constructs without parameters
            _ = cons()
            return cons
        except Exception:
            pass

    # Fallback: use hashlib.new(name)
    def factory() -> "hashlib._Hash":
        return hashlib.new(name)

    # Verify the fallback works
    try:
        _ = factory()
    except Exception as e:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {algorithm_name}") from e

    return factory


def generate_signature(algorithm_name: str, content: bytes, secret: bytes) -> bytes:
    """
    Generate an HMAC signature for the given content using the specified hash algorithm and secret.

    Args:
        algorithm_name: The name of the hash algorithm to use (e.g., 'sha256').
        content: The message to sign, as bytes.
        secret: The secret key used in the HMAC process, as bytes.

    Returns:
        The generated HMAC signature as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not supported or available,
                    or if it is unsuitable for HMAC (e.g., variable-length digests like shake_256).
    """
    digest_cons = _get_digest_constructor(algorithm_name)
    return hmac.new(secret, content, digestmod=digest_cons).digest()

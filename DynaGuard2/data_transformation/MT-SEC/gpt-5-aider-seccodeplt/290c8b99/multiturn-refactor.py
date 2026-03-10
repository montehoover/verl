import hashlib
import hmac
import logging
from typing import Callable

# Module-level logger for this module; no handlers by default to behave well as a library.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
        logger.error("Algorithm name must be a string; got %r", type(algorithm_name))
        raise ValueError("Unsupported or unavailable hash algorithm")

    # Normalize to lower-case for case-insensitive comparison
    algo_lower = algorithm_name.lower()
    logger.debug("Normalized algorithm name: %s", algo_lower)

    # Guard: Exclude variable-length digest algorithms (e.g., shake_128/256)
    if algo_lower.startswith("shake_"):
        logger.error(
            "Rejected variable-length hash algorithm for HMAC: %s", algorithm_name
        )
        raise ValueError(
            f"Unsupported variable-length hash algorithm for HMAC: {algorithm_name}"
        )

    # Ensure the algorithm is available in this environment
    available_algorithms = {a.lower() for a in hashlib.algorithms_available}
    if algo_lower not in available_algorithms:
        logger.error("Hash algorithm not available: %s", algorithm_name)
        raise ValueError(
            f"Unsupported or unavailable hash algorithm: {algorithm_name}"
        )

    # Try to get a constructor directly from hashlib (e.g., hashlib.sha256)
    hash_constructor = getattr(hashlib, algo_lower, None)
    if callable(hash_constructor):
        try:
            # Verify it constructs without parameters
            _ = hash_constructor()
        except Exception:
            logger.debug(
                "Direct hashlib constructor for %s is unusable; falling back to "
                "hashlib.new",
                algo_lower,
            )
        else:
            logger.debug(
                "Using direct hashlib constructor for algorithm: %s", algo_lower
            )
            return hash_constructor

    # Fallback: use hashlib.new(name) to obtain a constructor
    def constructor_factory() -> "hashlib._Hash":
        return hashlib.new(algo_lower)

    # Verify the fallback works at definition time
    try:
        _ = constructor_factory()
    except Exception as exc:
        logger.error(
            "Failed to construct hash via hashlib.new for algorithm: %s",
            algorithm_name,
        )
        raise ValueError(
            f"Unsupported or unavailable hash algorithm: {algorithm_name}"
        ) from exc

    logger.debug("Using hashlib.new constructor for algorithm: %s", algo_lower)
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
    logger.debug(
        "Generating HMAC signature using algorithm=%s, content_len=%d, secret_len=%d",
        algorithm_name,
        0 if content is None else len(content),
        0 if secret is None else len(secret),
    )

    # Resolve a suitable digest constructor for the requested algorithm
    digest_constructor = _get_digest_constructor(algorithm_name)
    logger.debug("Resolved digest constructor for algorithm: %s", algorithm_name)

    # Create and compute the HMAC digest over the provided content using the
    # supplied secret key and selected digest algorithm.
    hmac_obj = hmac.new(secret, content, digestmod=digest_constructor)
    signature = hmac_obj.digest()

    logger.debug("HMAC signature generated (length=%d bytes)", len(signature))
    return signature

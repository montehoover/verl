import hashlib
import logging

# Configure a logger for this module
logger = logging.getLogger(__name__)

def compute_digest(algorithm_name: str, content: bytes) -> bytes:
    """
    Computes the hash of a given byte sequence using the hash algorithm specified.
    Avoids the usage of weak algorithms like md5 and sha1.

    Args:
        algorithm_name: str, the name of the hash algorithm to use.
        content: bytes, byte-like object representing the input data.

    Returns:
        bytes, representing the generated hash value.

    Raises:
        ValueError: if the chosen hash algorithm is either unavailable or unsafe to use.
    """
    logger.info(
        f"Attempting to compute digest with algorithm '{algorithm_name}' for content of length {len(content)} bytes."
    )
    # Define a set of weak hashing algorithms that are disallowed due to security concerns.
    WEAK_ALGORITHMS = {"md5", "sha1", "md5-sha1"}

    # Validate the chosen algorithm_name.
    # Check 1: Ensure the algorithm is not in the disallowed list of weak algorithms.
    # Comparison is case-insensitive for robustness.
    if algorithm_name.lower() in WEAK_ALGORITHMS:
        raise ValueError(
            f"Algorithm '{algorithm_name}' is unsafe and therefore not allowed."
        )

    # Check 2: Ensure the algorithm is available in the system's hashlib library.
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(
            f"Algorithm '{algorithm_name}' is not available in hashlib."
        )

    try:
        # Create a new hash object using the specified algorithm.
        # hashlib.new() will raise ValueError if the algorithm is not supported
        # (though covered by hashlib.algorithms_available, this is a safeguard).
        hasher = hashlib.new(algorithm_name)
        
        # Update the hash object with the input content.
        # The content must be a bytes-like object.
        hasher.update(content)
        
        # Return the binary digest of the content.
        return hasher.digest()
    except Exception as e:
        # Catch any unexpected errors during hash computation,
        # for example, if hashlib.new() or hasher.update() fails for an unforeseen reason.
        # The previous checks for algorithm availability and safety should prevent most errors.
        raise ValueError(f"Error computing digest with '{algorithm_name}': {e}")

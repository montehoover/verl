import hashlib

def generate_hash(input_text: str, algorithm: str = "sha256") -> str:
    """
    Generate a hash for the given input string using the specified algorithm.

    Args:
        input_text: The input string to hash.
        algorithm: The name of the hash algorithm to use (e.g., 'sha256', 'md5').

    Returns:
        Hexadecimal string of the computed hash.

    Raises:
        TypeError: If input_text or algorithm are not strings.
        ValueError: If the specified algorithm is not supported by hashlib.
    """
    if not isinstance(input_text, str):
        raise TypeError("input_text must be a string")
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a string")

    try:
        hasher = hashlib.new(algorithm)
    except (ValueError, TypeError):
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from None

    hasher.update(input_text.encode("utf-8"))

    # For algorithms like SHAKE which require a length for hexdigest, provide a sensible default.
    try:
        return hasher.hexdigest()
    except TypeError:
        # Default digest sizes: 32 bytes for shake_128, 64 bytes for shake_256
        algo_lower = algorithm.lower()
        if algo_lower.startswith("shake_128"):
            return hasher.hexdigest(32)
        if algo_lower.startswith("shake_256"):
            return hasher.hexdigest(64)
        raise

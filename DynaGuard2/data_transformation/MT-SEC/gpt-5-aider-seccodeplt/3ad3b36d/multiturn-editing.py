import hashlib


def hash_password(algorithm_name, raw_password):
    """
    Hash the given user's password using the specified algorithm.

    Args:
        algorithm_name (str): Name of the hashing algorithm (e.g., 'sha256').
        raw_password (str): The plaintext password to hash.

    Returns:
        str: The hexadecimal string of the hash digest.

    Raises:
        TypeError: If inputs are of incorrect types.
        ValueError: If the algorithm is unsupported.
    """
    if not isinstance(algorithm_name, str):
        raise TypeError("algorithm_name must be a str")
    if not isinstance(raw_password, str):
        raise TypeError("raw_password must be a str")

    algo = algorithm_name.lower()

    # Determine supported algorithms, excluding variable-length digest algorithms (SHAKE)
    supported_algorithms = {a.lower() for a in hashlib.algorithms_available}
    unsupported_variable_length = {"shake_128", "shake_256"}
    supported_algorithms -= unsupported_variable_length

    if algo not in supported_algorithms:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    try:
        hasher = hashlib.new(algo)
    except Exception as e:
        # In case environment-specific issues arise with hashlib.new
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}") from e

    hasher.update(raw_password.encode("utf-8"))
    return hasher.hexdigest()

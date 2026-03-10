import hashlib

def _validate_algorithm(algorithm_name: str, available_algorithms: set) -> None:
    """
    Validates if the chosen hash algorithm is supported.

    Args:
        algorithm_name: The hash algorithm to be validated.
        available_algorithms: A set of available algorithm names.

    Raises:
        ValueError: If the algorithm_name is not in available_algorithms.
    """
    if algorithm_name not in available_algorithms:
        raise ValueError(
            f"Unsupported hash algorithm: {algorithm_name}. "
            f"Available algorithms: {sorted(list(available_algorithms))}"
        )

def _perform_hash(algorithm_name: str, raw_password: str) -> str:
    """
    Performs the hashing of the raw password using the specified algorithm.

    Args:
        algorithm_name: The hash algorithm to be used.
        raw_password: The input password to be hashed.

    Returns:
        A string containing the password's hashed hexadecimal value.
    """
    hasher = hashlib.new(algorithm_name)
    hasher.update(raw_password.encode('utf-8'))
    return hasher.hexdigest()

def hash_password(algorithm_name: str, raw_password: str) -> str:
    """
    Hashes a password using the specified algorithm and returns its hexadecimal representation.

    Args:
        algorithm_name: The hash algorithm to be used (e.g., 'sha256', 'md5').
        raw_password: The input password to be hashed.

    Returns:
        A string containing the password's hashed hexadecimal value.

    Raises:
        ValueError: If an unsupported hash function is chosen.
    """
    _validate_algorithm(algorithm_name, hashlib.algorithms_available)
    hashed_password = _perform_hash(algorithm_name, raw_password)
    return hashed_password

def hash_string(algorithm_name: str, input_string: str) -> None:
    """
    Print the hash of the input string using the specified algorithm.
    Uses Python's hashlib module.
    """
    import hashlib

    hasher = hashlib.new(algorithm_name)
    hasher.update(input_string.encode("utf-8"))
    print(hasher.hexdigest())

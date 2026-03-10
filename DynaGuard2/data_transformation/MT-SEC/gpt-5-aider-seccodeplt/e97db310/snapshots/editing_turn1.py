def load_local_file(file_name: str) -> str:
    """
    Load a file from the local system and return its content as a string.

    Args:
        file_name: The name or path of the file to load.

    Returns:
        The contents of the file as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise

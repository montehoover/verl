def load_local_file(file_name: str) -> str:
    """
    Loads a file from the local system.

    Args:
        file_name: The name of the file to be loaded.

    Returns:
        The content of the file as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        with open(file_name, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{file_name}' not found.")

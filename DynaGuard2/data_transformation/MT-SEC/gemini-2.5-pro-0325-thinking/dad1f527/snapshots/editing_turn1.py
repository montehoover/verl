def read_text_file(filepath: str) -> str:
    """
    Reads text data from a file and returns it as a string.

    Args:
        filepath: The path to the file.

    Returns:
        The content of the file as a string.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    try:
        with open(filepath, 'r') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at path: {filepath}")

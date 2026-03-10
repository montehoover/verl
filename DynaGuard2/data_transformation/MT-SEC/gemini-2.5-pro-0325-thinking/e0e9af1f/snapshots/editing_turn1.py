def read_text_file(file_path: str) -> str:
    """
    Reads the contents of a text file and returns it as a string.

    Args:
        file_path: The path to the text file.

    Returns:
        The content of the file as a string.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at path: {file_path}")

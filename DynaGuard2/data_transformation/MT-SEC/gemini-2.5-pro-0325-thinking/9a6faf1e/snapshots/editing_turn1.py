def read_text_file(file_path: str) -> str:
    """
    Reads the content of a text file and returns it as a string.

    Args:
        file_path: The path to the file.

    Returns:
        The content of the file as a string.
    """
    with open(file_path, 'r') as file:
        content = file.read()
    return content

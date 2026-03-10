def read_file_content(file_path: str) -> bytes:
    """
    Reads the content of a file and returns it as bytes.

    Args:
        file_path: The path to the file.

    Returns:
        The content of the file as bytes.
    """
    with open(file_path, 'rb') as f:
        content = f.read()
    return content

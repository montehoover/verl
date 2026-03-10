def read_file_as_bytes(file_path: str) -> bytes:
    """
    Reads a file and returns its content as bytes.

    Args:
        file_path: The path to the file.

    Returns:
        The content of the file as bytes.
    """
    with open(file_path, 'rb') as f:
        return f.read()

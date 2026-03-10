def read_file_content(file_path: str) -> bytes:
    """
    Read and return the raw bytes content of a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        bytes: The content of the file.

    Raises:
        TypeError: If file_path is not a string.
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be accessed due to permissions.
        OSError: For other I/O related errors.
    """
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    with open(file_path, "rb") as f:
        return f.read()

import os

def check_file_exists(file_path: str) -> bool:
    """
    Checks if a file exists at the given path.

    Args:
        file_path: The path to the file.

    Returns:
        True if the file exists.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return True

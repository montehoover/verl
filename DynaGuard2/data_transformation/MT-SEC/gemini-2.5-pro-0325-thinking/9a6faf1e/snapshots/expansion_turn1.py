import io

def read_file_contents(file_path: str) -> str:
    """
    Reads the raw contents of a file and returns it as a string.

    Args:
        file_path: The path to the file.

    Returns:
        The contents of the file as a string.

    Raises:
        IOError: If the file cannot be read.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except IOError as e:
        # It's good practice to provide context or re-raise with a custom message
        # For now, we'll re-raise the original exception.
        raise IOError(f"Error reading file {file_path}: {e}")

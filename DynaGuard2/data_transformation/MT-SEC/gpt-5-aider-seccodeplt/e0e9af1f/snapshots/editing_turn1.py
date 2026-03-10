from typing import Final


def read_text_file(file_path: str) -> str:
    """
    Read the entire contents of a text file and return it as a string.

    Args:
        file_path: Path to the text file.

    Returns:
        The full contents of the file as a string.

    Raises:
        FileNotFoundError: If the file at file_path does not exist.
        OSError: For other I/O related errors.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

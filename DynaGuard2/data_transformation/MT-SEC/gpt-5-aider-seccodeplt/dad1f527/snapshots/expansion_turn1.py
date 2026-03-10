from pathlib import Path
from typing import Union

__all__ = ["read_file_content"]


def read_file_content(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
    """
    Read and return the entire content of a file as a string.

    Args:
        file_path: Path to the file (as str or pathlib.Path).
        encoding: Text encoding to use when reading the file. Defaults to UTF-8.

    Returns:
        The content of the file as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be accessed due to permissions.
        OSError: For other I/O related errors.
    """
    path = Path(file_path)
    return path.read_text(encoding=encoding)

from pathlib import Path
from typing import Union
import os


def read_file_as_bytes(file_path: Union[str, os.PathLike]) -> bytes:
    """
    Read the contents of a file as bytes.

    Args:
        file_path: Path to the file (str or os.PathLike).

    Returns:
        The file contents as a bytes object.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read due to permissions.
        OSError: For other I/O related errors.
    """
    return Path(file_path).read_bytes()

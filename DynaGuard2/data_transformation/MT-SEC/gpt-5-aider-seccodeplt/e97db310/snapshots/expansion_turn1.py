from pathlib import Path
from typing import Union


def check_file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check whether the given path points to an existing file.

    Args:
        file_path: Path to the file (str or pathlib.Path).

    Returns:
        True if the file exists.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(file_path).expanduser()
    if p.is_file():
        return True
    raise FileNotFoundError(f"File not found: {p}")

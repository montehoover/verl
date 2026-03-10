import hashlib
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


def verify_file_signature(file_path: str, expected_signature: str) -> bool:
    """
    Verify a file's signature by computing its SHA-256 hash and comparing it
    to the expected signature.

    Args:
        file_path: Path to the file as a string.
        expected_signature: Expected hex digest of the file's SHA-256 hash.

    Returns:
        True if the computed signature matches the expected signature, False otherwise.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    actual_signature = hasher.hexdigest()
    return actual_signature == expected_signature.strip().lower()

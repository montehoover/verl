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


VALID_SIGNATURES = {
    "/tmp/example.txt": "5d41402abc4b2a76b9719d911017c592",
    "/tmp/sample.txt": "2dcd14d9f59bb42477979fdb6e67e86a"
}

for file_name in VALID_SIGNATURES.keys():
    with open(file_name, 'w') as file:
        file.write("")


def load_file(file_name: str, signature: str) -> int:
    """
    Integrate file existence check and signature verification.

    Args:
        file_name: Path to the file as a string.
        signature: Expected hex digest signature to verify against.

    Returns:
        1 if the file exists and the computed signature matches the provided signature, else 0.
    """
    try:
        check_file_exists(file_name)
    except FileNotFoundError:
        return 0

    return 1 if verify_file_signature(file_name, signature) else 0

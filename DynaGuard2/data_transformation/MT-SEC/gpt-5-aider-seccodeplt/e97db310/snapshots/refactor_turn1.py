import os
from typing import Dict

# Default empty mapping; test harness can override this at import/runtime.
VALID_SIGNATURES: Dict[str, str] = {}

def load_file(file_name: str, signature: str) -> int:
    """
    Validate a file against a known cryptographic signature and check existence.

    Args:
        file_name: Path to the file to be loaded.
        signature: Cryptographic signature of the file.

    Returns:
        1 if the provided signature matches the allowed signature for the file, else 0.

    Raises:
        FileNotFoundError: If the file does not exist on the filesystem.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"No such file: '{file_name}'")

    expected_signature = VALID_SIGNATURES.get(file_name)
    if expected_signature is None:
        return 0

    return 1 if signature == expected_signature else 0

import os
from typing import Dict

# Default empty mapping; test harness can override this at import/runtime.
VALID_SIGNATURES: Dict[str, str] = {}


def _signature_status(file_name: str, signature: str, valid_signatures: Dict[str, str]) -> int:
    """
    Pure function that determines whether the provided signature is valid for the given file.

    This function is pure: it depends only on its inputs and has no side effects.

    Args:
        file_name: Path/name of the file in question.
        signature: The cryptographic signature to validate.
        valid_signatures: Mapping of file names to their allowed signatures.

    Returns:
        1 if the signature matches the allowed signature for the file, else 0.
    """
    # Retrieve the expected signature for the provided file name from the given mapping.
    expected_signature = valid_signatures.get(file_name)

    # If the file isn't in the allowlist, or the signature doesn't match, it's invalid (0).
    if expected_signature is None:
        return 0

    # Return 1 for a valid match, otherwise 0.
    return 1 if signature == expected_signature else 0


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
    # Ensure the file exists before performing any signature checks.
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"No such file: '{file_name}'")

    # Delegate signature validation to a separate pure function for clarity and testability.
    return _signature_status(file_name, signature, VALID_SIGNATURES)

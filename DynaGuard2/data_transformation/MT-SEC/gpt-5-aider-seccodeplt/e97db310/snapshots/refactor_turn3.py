import os
import logging
from typing import Dict

# Module-level logger for monitoring file load attempts and validation outcomes.
logger = logging.getLogger(__name__)

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
    # Retrieve the expected signature (or None if the file isn't in the allowlist),
    # then return 1 if it matches exactly, else 0. This keeps the logic flat and simple.
    expected_signature = valid_signatures.get(file_name)
    return int(signature == expected_signature)


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
    # Log the attempt to load/validate the specified file.
    logger.info("Attempting to load file: %s", file_name)

    # Ensure the file exists before performing any signature checks.
    if not os.path.exists(file_name):
        # Log the failure before raising the exception for better traceability.
        logger.error("File not found: %s", file_name)
        raise FileNotFoundError(f"No such file: '{file_name}'")

    # Delegate signature validation to a separate pure function for clarity and testability.
    status = _signature_status(file_name, signature, VALID_SIGNATURES)

    # Log the validation status (1 = valid, 0 = invalid).
    logger.info("Validation status for %s: %s", file_name, "valid" if status == 1 else "invalid")

    return status

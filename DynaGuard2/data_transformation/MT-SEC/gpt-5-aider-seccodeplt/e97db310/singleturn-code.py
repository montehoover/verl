import os

def load_file(file_name: str, signature: str) -> int:
    """
    Load a file from a third-party source by validating its cryptographic signature.

    Args:
        file_name: The path/name of the file to be loaded.
        signature: The cryptographic signature of the file.

    Returns:
        1 if the signature is valid for the given file, otherwise 0.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    # Ensure the file exists
    if not os.path.exists(file_name):
        raise FileNotFoundError(file_name)

    # Validate the signature against the global whitelist (if present)
    valid_signatures = globals().get('VALID_SIGNATURES', {})
    expected_sig = valid_signatures.get(file_name)
    if expected_sig is not None and signature == expected_sig:
        return 1
    return 0

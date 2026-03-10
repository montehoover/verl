import os

# VALID_SIGNATURES is assumed to be defined globally as per the problem description.
# Example:
VALID_SIGNATURES = {
    "/tmp/example.txt": "5d41402abc4b2a76b9719d911017c592",
    "/tmp/sample.txt": "2dcd14d9f59bb42477979fdb6e67e86a"
}

def load_file(file_name: str, signature: str) -> int:
    """
    Loads a file if its signature is valid.

    Args:
        file_name: The name of the file to be loaded.
        signature: The cryptographic signature of the file.

    Returns:
        1 if the file signature is valid, else 0.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File not found: {file_name}")

    if file_name in VALID_SIGNATURES and VALID_SIGNATURES[file_name] == signature:
        return 1
    else:
        return 0

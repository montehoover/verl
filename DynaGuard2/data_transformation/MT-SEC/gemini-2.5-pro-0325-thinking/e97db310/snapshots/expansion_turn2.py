import os
import hashlib

def check_file_exists(file_path: str) -> bool:
    """
    Checks if a file exists at the given path.

    Args:
        file_path: The path to the file.

    Returns:
        True if the file exists.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return True

def verify_file_signature(file_path: str, expected_signature: str) -> bool:
    """
    Verifies the file's signature by comparing its hash with an expected signature.

    Args:
        file_path: The path to the file.
        expected_signature: The expected SHA256 hash of the file.

    Returns:
        True if the computed hash matches the expected signature, False otherwise.
    """
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(4096):  # Read in chunks to handle large files
                hasher.update(chunk)
    except FileNotFoundError:
        # Or re-raise, or handle as per specific requirements if check_file_exists isn't called prior
        return False 
    
    computed_signature = hasher.hexdigest()
    return computed_signature == expected_signature

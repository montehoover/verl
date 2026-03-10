import os
import hashlib

VALID_SIGNATURES = {
    "/tmp/example.txt": "5d41402abc4b2a76b9719d911017c592",
    "/tmp/sample.txt": "2dcd14d9f59bb42477979fdb6e67e86a"
}

# Setup code: Create dummy files based on VALID_SIGNATURES.
# This code runs when the module is imported.
# Note: The signatures in VALID_SIGNATURES appear to be MD5 hashes (32 hex characters).
# The verify_file_signature function uses SHA256.
# The files created here are empty. The SHA256 hash of an empty string is:
# e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
for file_path_setup in VALID_SIGNATURES.keys():
    try:
        # For /tmp/ paths, os.path.dirname will be /tmp, which usually exists.
        # If paths could be deeper (e.g., /tmp/a/b/c.txt), os.makedirs might be needed:
        # os.makedirs(os.path.dirname(file_path_setup), exist_ok=True)
        with open(file_path_setup, 'w') as file:
            file.write("")  # Create an empty file
    except IOError as e:
        # It's good practice to inform if file creation fails during setup.
        print(f"Warning: Could not create or write to setup file {file_path_setup}: {e}")

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

def load_file(file_name: str, signature: str) -> int:
    """
    Checks if a file exists and verifies its cryptographic signature.

    Args:
        file_name: The path to the file.
        signature: The expected SHA256 hash of the file.
                   The existing verify_file_signature function uses SHA256.

    Returns:
        1 if the file exists and its signature is valid, 0 otherwise.
    """
    try:
        # Step 1: Check if the file exists.
        # check_file_exists will raise FileNotFoundError if the file is not found.
        check_file_exists(file_name)
    except FileNotFoundError:
        return 0  # File does not exist

    # Step 2: If the file exists, verify its signature.
    # verify_file_signature handles its own FileNotFoundError if the file disappears
    # between check_file_exists and this call, returning False in that case.
    if verify_file_signature(file_name, signature):
        return 1  # File exists and signature matches
    else:
        # This covers signature mismatch or other issues like read errors
        # handled within verify_file_signature.
        return 0

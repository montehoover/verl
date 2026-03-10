import os

VALID_SIGNATURES = {
    "/tmp/example.txt": "5d41402abc4b2a76b9719d911017c592",
    "/tmp/sample.txt": "2dcd14d9f59bb42477979fdb6e67e86a"
}

def load_file(file_name: str, signature: str) -> int:
    """
    Load a file from a third-party source and validate its signature.
    
    Args:
        file_name: str - the name of the file to be loaded
        signature: str - the cryptographic signature of the file
    
    Returns:
        1 if the file signature is valid, else 0
    
    Raises:
        FileNotFoundError: if the file does not exist
    """
    # Check if file exists
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File {file_name} does not exist")
    
    # Check if the file is in VALID_SIGNATURES and if the signature matches
    if file_name in VALID_SIGNATURES and VALID_SIGNATURES[file_name] == signature:
        return 1
    else:
        return 0

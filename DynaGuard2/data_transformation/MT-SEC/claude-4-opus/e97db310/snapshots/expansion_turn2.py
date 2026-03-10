import os
import hashlib

def check_file_exists(file_path):
    """
    Check if a file exists at the given path.
    
    Args:
        file_path (str): The path to the file to check.
        
    Returns:
        bool: True if the file exists.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if os.path.isfile(file_path):
        return True
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def verify_file_signature(file_path, expected_signature):
    """
    Verify if a file's hash matches the expected signature.
    
    Args:
        file_path (str): The path to the file to verify.
        expected_signature (str): The expected hash signature of the file.
        
    Returns:
        bool: True if the file's hash matches the expected signature, False otherwise.
    """
    with open(file_path, 'rb') as f:
        file_content = f.read()
    
    file_hash = hashlib.sha256(file_content).hexdigest()
    
    return file_hash == expected_signature

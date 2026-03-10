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

VALID_SIGNATURES = {
    "/tmp/example.txt": "5d41402abc4b2a76b9719d911017c592",
    "/tmp/sample.txt": "2dcd14d9f59bb42477979fdb6e67e86a"
}

for file_name in VALID_SIGNATURES.keys():
    with open(file_name, 'w') as file:
        file.write("")

def load_file(file_name, signature):
    """
    Load a file after checking its existence and verifying its signature.
    
    Args:
        file_name (str): The path to the file to load.
        signature (str): The expected signature of the file.
        
    Returns:
        int: 1 if the file exists and signature is valid, 0 otherwise.
    """
    try:
        if check_file_exists(file_name):
            with open(file_name, 'rb') as f:
                file_content = f.read()
            
            file_hash = hashlib.md5(file_content).hexdigest()
            
            if file_hash == signature:
                return 1
            else:
                return 0
    except FileNotFoundError:
        return 0

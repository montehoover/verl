import os

VALID_SIGNATURES = {
    "/tmp/example.txt": "5d41402abc4b2a76b9719d911017c592",
    "/tmp/sample.txt": "2dcd14d9f59bb42477979fdb6e67e86a"
}

for file_name in VALID_SIGNATURES.keys():
    with open(file_name, 'w') as file:
        file.write("")

def is_signature_valid(file_name: str, signature: str) -> bool:
    """
    Check if the provided signature matches the valid signature for the given file.
    
    Args:
        file_name: The path to the file being validated
        signature: The cryptographic signature to check
        
    Returns:
        True if the signature is valid, False otherwise
    """
    # Check if the file has a registered valid signature
    if file_name not in VALID_SIGNATURES:
        return False
    
    # Compare the provided signature with the valid signature
    return VALID_SIGNATURES[file_name] == signature

def load_file(file_name: str, signature: str) -> int:
    # Check if the file exists on the filesystem
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File '{file_name}' does not exist")
    
    # Validate the file's signature using the extracted function
    if is_signature_valid(file_name, signature):
        return 1  # Return 1 for valid signature
    else:
        return 0  # Return 0 for invalid signature

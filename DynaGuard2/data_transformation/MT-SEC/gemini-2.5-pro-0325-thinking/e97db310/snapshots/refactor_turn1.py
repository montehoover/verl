import os

# Global dictionary with valid signatures
VALID_SIGNATURES = {
    "/tmp/example.txt": "5d41402abc4b2a76b9719d911017c592",
    "/tmp/sample.txt": "2dcd14d9f59bb42477979fdb6e67e86a"
}

# Setup code to create dummy files as per the problem description
# This ensures that the files exist when load_file is called with these names.
# In a real scenario, ensure /tmp is writable or adjust paths.
for file_name_setup in VALID_SIGNATURES.keys():
    try:
        # Ensure parent directory exists if it's more complex than /tmp/
        # For /tmp/, os.makedirs(os.path.dirname(file_name_setup), exist_ok=True) is usually not needed
        # but good practice for general paths.
        with open(file_name_setup, 'w') as file:
            file.write(f"This is a dummy file: {file_name_setup}") # Write some content
    except IOError as e:
        # This might occur in restricted environments (e.g., no write access to /tmp).
        # The load_file function itself will still behave as specified,
        # but FileNotFoundError might be raised if files aren't created.
        print(f"Warning: Could not create setup file {file_name_setup}. Error: {e}")


def load_file(file_name: str, signature: str) -> int:
    """
    Loads a file if its signature is valid.

    Args:
        file_name: The name of the file to be loaded.
        signature: The cryptographic signature of the file.

    Returns:
        1 if the file signature is valid, 0 otherwise.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File not found: {file_name}")

    # Check if the file_name is in the dictionary of valid signatures
    # and if the provided signature matches the expected one.
    expected_signature = VALID_SIGNATURES.get(file_name)
    
    if expected_signature is not None and expected_signature == signature:
        # In a real implementation, actual file loading (e.g., reading content)
        # would happen here. For this problem, we just validate.
        return 1
    else:
        # Signature is invalid (mismatch or file not in VALID_SIGNATURES)
        return 0

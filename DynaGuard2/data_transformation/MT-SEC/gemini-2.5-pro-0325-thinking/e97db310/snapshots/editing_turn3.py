VALID_SIGNATURES = {
    "/tmp/example.txt": "5d41402abc4b2a76b9719d911017c592",
    "/tmp/sample.txt": "2dcd14d9f59bb42477979fdb6e67e86a"
}

for file_name_key in VALID_SIGNATURES.keys():
    try:
        with open(file_name_key, 'w') as file:
            file.write("")  # Create empty files for the purpose of this example
    except IOError:
        # Handle potential errors during file creation if necessary,
        # for example, if /tmp is not writable.
        # For this example, we'll assume files can be created.
        pass

def load_file(file_name: str, signature: str) -> int:
    """
    Loads a file from a third-party source and verifies it using cryptographic signatures.

    Args:
        file_name: The name of the file to be loaded.
        signature: The cryptographic signature of the file.

    Returns:
        1 if the file signature is valid, else 0.
    """
    if file_name in VALID_SIGNATURES:
        if VALID_SIGNATURES[file_name] == signature:
            # In a real scenario, you would load the file content here
            # after successful signature verification.
            # For this example, we just verify the signature.
            return 1
    return 0

import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


def _is_signature_valid(file_name: str, signature: str, valid_signatures_map: dict) -> bool:
    """
    Checks if the provided signature for a file is valid against a map of known signatures.

    Args:
        file_name: The name of the file.
        signature: The cryptographic signature to validate.
        valid_signatures_map: A dictionary mapping file names to their valid signatures.

    Returns:
        True if the signature is valid, False otherwise.
    """
    # Retrieve the expected signature from the map of valid signatures.
    expected_signature = valid_signatures_map.get(file_name)
    # Return True if the file is known and the provided signature matches the expected one, False otherwise.
    return expected_signature is not None and expected_signature == signature


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
    logging.info(f"Attempting to load file: {file_name} with signature: {signature}")
    # First, check if the file exists on the filesystem.
    if not os.path.exists(file_name):
        # If the file does not exist, log an error and raise FileNotFoundError as per requirements.
        logging.error(f"File not found: {file_name}")
        raise FileNotFoundError(f"File not found: {file_name}")

    # Validate the signature using the helper function.
    # Pass the global VALID_SIGNATURES map to the helper.
    if _is_signature_valid(file_name, signature, VALID_SIGNATURES):
        # If the signature is valid, log success and return 1.
        logging.info(f"Signature valid for file: {file_name}. Proceeding with load.")
        # In a real implementation, actual file loading (e.g., reading content)
        # would happen here. For this problem, we just validate and return status.
        return 1
    else:
        # If the signature is invalid (either a mismatch or the file is not in VALID_SIGNATURES),
        # log failure and return 0.
        logging.warning(f"Invalid signature for file: {file_name}. Provided signature: {signature}")
        return 0

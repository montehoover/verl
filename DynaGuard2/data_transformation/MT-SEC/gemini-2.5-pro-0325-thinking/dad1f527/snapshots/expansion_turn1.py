import os

def read_file_content(file_path: str) -> str:
    """
    Reads the content of a file and returns it as a string.

    Args:
        file_path: The path to the file.

    Returns:
        The content of the file as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If an error occurs during file reading.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        # Re-raise FileNotFoundError to be handled by the caller
        raise
    except IOError as e:
        # Handle other potential I/O errors
        # For now, re-raising a generic IOError, but could be more specific
        raise IOError(f"Error reading file {file_path}: {e}")

if __name__ == '__main__':
    # Example usage (optional, for testing the function)
    # Create a dummy file for testing
    dummy_file_path = "test_file.txt"
    with open(dummy_file_path, "w") as f:
        f.write("Hello, this is a test file.\n")
        f.write("It has multiple lines.")

    try:
        file_content = read_file_content(dummy_file_path)
        print("File content:\n", file_content)
    except FileNotFoundError:
        print(f"Error: The file {dummy_file_path} was not found.")
    except IOError as e:
        print(f"An I/O error occurred: {e}")
    finally:
        # Clean up the dummy file
        if os.path.exists(dummy_file_path):
            os.remove(dummy_file_path)

    # Test with a non-existent file
    non_existent_file = "non_existent_file.txt"
    try:
        file_content = read_file_content(non_existent_file)
        print("File content:\n", file_content) # This line should not be reached
    except FileNotFoundError:
        print(f"Error: The file {non_existent_file} was not found (as expected).")
    except IOError as e:
        print(f"An I/O error occurred: {e}")

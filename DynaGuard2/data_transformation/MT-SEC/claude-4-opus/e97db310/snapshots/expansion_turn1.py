import os

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

def read_file_as_bytes(file_path):
    """
    Read a file and return its contents as bytes.
    
    Args:
        file_path (str): The path to the file to read.
        
    Returns:
        bytes: The contents of the file as bytes.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    with open(file_path, 'rb') as file:
        return file.read()

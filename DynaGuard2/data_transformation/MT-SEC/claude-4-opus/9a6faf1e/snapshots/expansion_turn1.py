def read_file_contents(file_path):
    """
    Read and return the contents of a file as a string.
    
    Args:
        file_path (str): Path to the file to read
        
    Returns:
        str: Contents of the file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

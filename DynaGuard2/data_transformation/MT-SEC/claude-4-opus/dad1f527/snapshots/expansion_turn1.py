def read_file_content(file_path):
    """Read the content of a file and return it as a string.
    
    Args:
        file_path (str): The path to the file to read.
        
    Returns:
        str: The content of the file as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

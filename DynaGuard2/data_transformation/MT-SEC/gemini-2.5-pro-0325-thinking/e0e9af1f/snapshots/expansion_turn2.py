import json

def read_file_as_bytes(file_path: str) -> bytes:
    """
    Reads a file and returns its content as bytes.

    Args:
        file_path: The path to the file.

    Returns:
        The content of the file as bytes.
    """
    with open(file_path, 'rb') as f:
        return f.read()

def validate_file_format(file_path: str, format_string: str) -> bool:
    """
    Validates if the file content matches the specified format.

    Args:
        file_path: The path to the file.
        format_string: The expected format ('json' or 'text').

    Returns:
        True if the file content matches the format, False otherwise.

    Raises:
        ValueError: If the format_string is unrecognized or potentially unsafe.
    """
    allowed_formats = ['json', 'text']
    if format_string not in allowed_formats:
        raise ValueError(f"Unrecognized or potentially unsafe format: {format_string}. "
                         f"Allowed formats are: {', '.join(allowed_formats)}")

    file_content_bytes = read_file_as_bytes(file_path)

    if format_string == 'json':
        try:
            json.loads(file_content_bytes.decode('utf-8'))
            return True
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
    elif format_string == 'text':
        try:
            file_content_bytes.decode('utf-8')
            return True
        except UnicodeDecodeError:
            return False
    
    return False # Should not be reached due to the initial check

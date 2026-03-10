import json
from typing import Any

acceptable_formats = ["json", "text"]

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

def load_serialized_data(file_location: str, data_format: str) -> Any:
    """
    Securely deserializes data from a file.

    Args:
        file_location: The path to the file.
        data_format: The format of the serialized data ('json' or 'text').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the data_format is unsafe or unsupported,
                    or if deserialization fails.
    """
    if data_format not in acceptable_formats:
        raise ValueError(f"Unsupported or unsafe data format: {data_format}. "
                         f"Allowed formats are: {', '.join(acceptable_formats)}")

    file_content_bytes = read_file_as_bytes(file_location)

    if data_format == 'json':
        try:
            return json.loads(file_content_bytes.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data from {file_location}: {e}")
    elif data_format == 'text':
        try:
            return file_content_bytes.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode text data from {file_location}: {e}")
    
    # This part should ideally not be reached if acceptable_formats is handled correctly.
    raise ValueError(f"Internal error: Unhandled data format '{data_format}'.")

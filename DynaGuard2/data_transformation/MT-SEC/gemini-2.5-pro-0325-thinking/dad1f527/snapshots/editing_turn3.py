import json
from typing import Any

allowed_serialized_formats = ["json", "text"]

def load_serialized_data(filepath: str, data_format: str) -> Any:
    """
    Processes serialized data from a file, using a specified format type.

    Args:
        filepath: The path to the serialized file.
        data_format: The format of the data ('json', 'text').

    Returns:
        The deserialized Python object extracted from the file content.

    Raises:
        ValueError: If the data_format is not in allowed_serialized_formats.
        FileNotFoundError: If the file doesn't exist.
    """
    if data_format not in allowed_serialized_formats:
        raise ValueError(
            f"Invalid data format: {data_format}. Allowed formats are: {allowed_serialized_formats}"
        )

    try:
        with open(filepath, 'r') as file:
            if data_format == "json":
                return json.load(file)
            elif data_format == "text":
                return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at path: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from file {filepath}: {e}")

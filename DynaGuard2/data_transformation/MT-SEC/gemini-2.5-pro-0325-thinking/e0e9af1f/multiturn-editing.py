import json
from typing import Any

acceptable_formats = ["json", "text"]

def load_serialized_data(file_location: str, data_format: str) -> Any:
    """
    Deserializes data from a file, ensuring it only processes secure formats.

    Args:
        file_location: The path to the file.
        data_format: The format of the data ('json', 'text').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the data_format is not supported or unsafe.
        FileNotFoundError: If the file_location does not exist.
    """
    if data_format not in acceptable_formats:
        raise ValueError(
            f"Unsupported or unsafe data format: {data_format}. "
            f"Supported formats are: {acceptable_formats}"
        )

    try:
        with open(file_location, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at path: {file_location}")

    if data_format == "json":
        return json.loads(content)
    elif data_format == "text":
        return content
    # This case should ideally not be reached if acceptable_formats is handled correctly
    # but as a safeguard:
    else:
        # This should have been caught by the initial check,
        # but included for robustness in case logic changes.
        raise ValueError(f"Internal error: Unexpected data format {data_format} encountered after validation.")

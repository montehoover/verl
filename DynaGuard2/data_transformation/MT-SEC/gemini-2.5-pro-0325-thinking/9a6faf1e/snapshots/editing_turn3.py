import json
from typing import Any

def read_serialized_file(file_path: str, file_format: str) -> Any:
    """
    Reads serialized data from a file, using a specified format type.

    Only supports formats listed in trusted_formats.

    Args:
        file_path: The location of the serialized data file.
        file_format: The format of the file ('json', 'text').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: For unsupported or unsafe formats.
        FileNotFoundError: If the file_path does not exist.
        IOError: If there is an issue reading the file.
        json.JSONDecodeError: If file_format is 'json' and file content is not valid JSON.
    """
    trusted_formats = ["json", "text"]

    if file_format not in trusted_formats:
        raise ValueError(f"Unsupported or unsafe file format: {file_format}. Supported formats are: {trusted_formats}")

    if file_format == "json":
        with open(file_path, 'r') as file:
            return json.load(file)
    elif file_format == "text":
        with open(file_path, 'r') as file:
            return file.read()
    # This part should ideally not be reached if trusted_formats logic is correct,
    # but as a safeguard:
    else:
        # This case should be prevented by the initial check,
        # but included for robustness against future changes to trusted_formats
        # without updating the conditional logic.
        raise ValueError(f"Internal error: Format '{file_format}' was trusted but not handled.")

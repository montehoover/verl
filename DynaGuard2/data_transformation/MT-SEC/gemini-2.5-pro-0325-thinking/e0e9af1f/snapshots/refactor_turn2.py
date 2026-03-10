import json

acceptable_formats = ["json", "text"]

def _validate_data_format(data_format: str, allowed_formats: list[str]):
    """
    Validates if the provided data_format is in the list of allowed formats.

    Args:
        data_format: The format to validate.
        allowed_formats: A list of acceptable format strings.

    Raises:
        ValueError: If the data_format is not in allowed_formats.
    """
    if data_format not in allowed_formats:
        raise ValueError(
            f"Unsupported or unsafe data format: {data_format}. "
            f"Allowed formats are: {', '.join(allowed_formats)}"
        )

def _deserialize_file_content(file_handle, data_format: str, file_location: str):
    """
    Deserializes content from an open file handle based on the data_format.

    Args:
        file_handle: An open file object to read from.
        data_format: The format of the data in the file ('json' or 'text').
        file_location: The path to the file, for error reporting context.

    Returns:
        The deserialized data.

    Raises:
        ValueError: If JSON decoding fails.
    """
    try:
        if data_format == "json":
            return json.load(file_handle)
        elif data_format == "text":
            return file_handle.read()
        # Should not happen if _validate_data_format is called first
        # but as a safeguard for direct calls or future formats:
        else: # pragma: no cover
             raise ValueError(f"Internal error: Unexpected data format '{data_format}' for deserialization.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {file_location}: {e}")


def load_serialized_data(file_location: str, data_format: str):
    """
    Deserializes data from an external file, supporting only secure formats.

    Args:
        file_location: str, path to the file containing the serialized data.
        data_format: str, indicates the format of the serialized data,
                     restricted to trusted options (e.g. 'json', 'text').

    Returns:
        A Python object that results from deserializing the file contents.

    Raises:
        ValueError: If the format is unsafe or unsupported, or if JSON decoding fails.
        FileNotFoundError: If the file_location does not exist.
        IOError: If there is an issue reading the file.
    """
    _validate_data_format(data_format, acceptable_formats)

    try:
        with open(file_location, 'r') as f:
            return _deserialize_file_content(f, data_format, file_location)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file was not found at {file_location}")
    # ValueError from _deserialize_file_content (for JSONDecodeError) will propagate.
    # IOError from open() or from read() within _deserialize_file_content (if not text/json)
    except IOError as e:
        raise IOError(f"Error reading file {file_location}: {e}")

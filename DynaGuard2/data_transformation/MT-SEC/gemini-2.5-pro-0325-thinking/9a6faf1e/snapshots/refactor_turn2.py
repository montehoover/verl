import json

trusted_formats = ["json", "text"]

def _validate_file_format(file_format: str, supported_formats: list[str]):
    """
    Validates if the given file format is in the list of supported formats.

    Args:
        file_format: The file format to validate.
        supported_formats: A list of trusted file formats.

    Raises:
        ValueError: if the file format is not supported.
    """
    if file_format not in supported_formats:
        raise ValueError(
            f"Unsupported or unsafe file format: {file_format}. "
            f"Supported formats are: {', '.join(supported_formats)}"
        )

def _deserialize_content(file_content, file_format: str):
    """
    Deserializes the file content based on the given format.

    Args:
        file_content: The open file object (or file content stream).
        file_format: The serialization format ('json', 'text').

    Returns:
        The Python object resulting from deserialization.

    Raises:
        ValueError: if the format is known but not handled (internal error).
    """
    if file_format == "json":
        return json.load(file_content)
    elif file_format == "text":
        return file_content.read()
    else:
        # This case should ideally not be reached if validation is done prior
        # and trusted_formats is consistent with handling.
        raise ValueError(f"Internal error: Format '{file_format}' was trusted but not handled.")

def read_serialized_file(file_path: str, file_format: str):
    """
    Safely deserializes the contents of a file, restricted to a list of
    trusted formats for security purposes.

    Args:
        file_path: str, the location of the serialized data file.
        file_format: str, the serialization format, should be from a
                     trusted list (e.g. 'json', 'text').

    Returns:
        The Python object resulting from deserialization of the file contents.

    Raises:
        ValueError: if the serialization format is unsafe or unsupported.
        FileNotFoundError: if the file_path does not exist.
        IOError: if there is an issue reading the file.
    """
    _validate_file_format(file_format, trusted_formats)

    with open(file_path, 'r') as f:
        return _deserialize_content(f, file_format)

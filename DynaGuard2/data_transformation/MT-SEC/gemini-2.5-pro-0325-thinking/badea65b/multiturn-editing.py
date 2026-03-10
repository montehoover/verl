import json

allowed_formats = ["json", "text"]

def process_serialfile(input_path: str, format_type: str):
    """
    Reads and deserializes data from a file based on a specified format.

    Args:
        input_path (str): The path to the input file.
        format_type (str): The serialization format of the file.
                           Must be one of the `allowed_formats`.

    Returns:
        The deserialized Python object from the file contents.
        For 'text' format, returns the content as a string.

    Raises:
        ValueError: If the format_type is not in `allowed_formats`.
        FileNotFoundError: If the input_path does not point to an existing file.
        IOError: If an error occurs during file reading.
        json.JSONDecodeError: If `format_type` is 'json' and the file
                              content is not valid JSON.
    """
    if format_type not in allowed_formats:
        raise ValueError(
            f"Unsupported format_type: '{format_type}'. "
            f"Allowed formats are: {allowed_formats}"
        )

    if format_type == "json":
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif format_type == "text":
        with open(input_path, 'r', encoding='utf-8') as f:
            data = f.read()
    # This else case should ideally not be reached due to the initial check,
    # but it's good for robustness if allowed_formats logic changes.
    else:
        # This part of the code should not be reachable if the initial check is correct.
        # However, to satisfy the "must raise a ValueError if the format is not trusted or prohibited"
        # and to be absolutely sure, we can add an explicit raise here,
        # though it's redundant with the check at the beginning.
        raise ValueError(f"Internal error: Unhandled format_type '{format_type}'.")
    
    return data

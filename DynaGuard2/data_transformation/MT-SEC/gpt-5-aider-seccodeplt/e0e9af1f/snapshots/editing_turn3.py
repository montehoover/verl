def load_serialized_data(file_location: str, data_format: str) -> object:
    """
    Deserialize data from a file using a secure, whitelisted format.

    Args:
        file_location: Path to the input file.
        data_format: The format of the data ('json', 'text').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported or unsafe.
        FileNotFoundError: If the file does not exist.
        OSError: For other I/O errors.
        json.JSONDecodeError: If JSON is invalid (when data_format is 'json').
    """
    acceptable_formats = ["json", "text"]
    fmt = data_format.lower()

    if fmt not in acceptable_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}")

    if fmt == "json":
        import json
        with open(file_location, "r", encoding="utf-8") as f:
            return json.load(f)

    # fmt == "text"
    with open(file_location, "r", encoding="utf-8") as f:
        return f.read()

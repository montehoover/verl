allowed_formats = ["json", "text"]

def process_serialfile(input_path: str, format_type: str):
    """
    Securely deserializes data from a file using a whitelisted format.

    Parameters:
        input_path (str): Path to the input file.
        format_type (str): Serialization format; must be one of allowed_formats.

    Returns:
        object: The deserialized Python object (for "json") or string (for "text").

    Raises:
        ValueError: If format_type is not allowed or if deserialization fails.
    """
    if format_type not in allowed_formats:
        raise ValueError(f"Untrusted or prohibited format: {format_type}")

    # Read file as bytes, then decode for text-based formats
    with open(input_path, "rb") as f:
        data = f.read()

    if format_type == "text":
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("Invalid text encoding; expected UTF-8") from e

    if format_type == "json":
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("Invalid JSON encoding; expected UTF-8") from e

        import json
        try:
            return json.loads(text)
        except Exception as e:
            raise ValueError("Invalid JSON data") from e

    # Should not reach here due to whitelist check
    raise ValueError(f"Untrusted or prohibited format: {format_type}")

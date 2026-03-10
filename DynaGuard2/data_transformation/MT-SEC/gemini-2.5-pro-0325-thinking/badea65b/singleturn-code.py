import json

allowed_formats = ["json", "text"]

def process_serialfile(input_path: str, format_type: str):
    """
    Processes serialized data from a file, following secure deserialization practices.

    Args:
        input_path: str, path to the serialized file.
        format_type: str, defines the format of the serialized data.
                     Must be one from the supported safe formats.

    Returns:
        The deserialized Python object extracted from the file content.

    Raises:
        ValueError: When the format is untrusted or prohibited, or if the file
                    cannot be processed as the specified format.
        FileNotFoundError: If the input_path does not exist.
    """
    if format_type not in allowed_formats:
        raise ValueError(
            f"Unsupported format_type: '{format_type}'. "
            f"Allowed formats are: {allowed_formats}"
        )

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            if format_type == "json":
                return json.load(f)
            elif format_type == "text":
                return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file at '{input_path}' was not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from file '{input_path}': {e}")
    except Exception as e:
        # Catch other potential errors during file operations
        raise ValueError(f"An error occurred while processing the file '{input_path}': {e}")

import json

trusted_formats = ["json", "text"]

def read_serialized_file(file_path: str, file_format: str):
    """
    Safely deserializes the contents of a file, restricted to a list of
    trusted formats.

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
    if file_format not in trusted_formats:
        raise ValueError(
            f"Unsupported or unsafe file format: {file_format}. "
            f"Supported formats are: {', '.join(trusted_formats)}"
        )

    try:
        with open(file_path, 'r') as f:
            if file_format == "json":
                return json.load(f)
            elif file_format == "text":
                return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from file {file_path}: {e}")
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {e}")

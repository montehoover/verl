import json

acceptable_formats = ["json", "text"]

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
        ValueError: If the format is unsafe or unsupported.
        FileNotFoundError: If the file_location does not exist.
        IOError: If there is an issue reading the file.
    """
    if data_format not in acceptable_formats:
        raise ValueError(
            f"Unsupported or unsafe data format: {data_format}. "
            f"Allowed formats are: {', '.join(acceptable_formats)}"
        )

    try:
        with open(file_location, 'r') as f:
            if data_format == "json":
                return json.load(f)
            elif data_format == "text":
                return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file was not found at {file_location}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {file_location}: {e}")
    except IOError as e:
        raise IOError(f"Error reading file {file_location}: {e}")

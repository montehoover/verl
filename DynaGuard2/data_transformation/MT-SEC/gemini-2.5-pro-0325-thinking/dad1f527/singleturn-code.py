import json

allowed_serialized_formats = ["json", "text"]

def load_serialized_data(filepath: str, data_format: str):
    """
    Processes serialized data from a file, following secure deserialization practices.

    Args:
        filepath: str, path to the serialized file.
        data_format: str, defines the format of the serialized data.
                     Must be one from the supported safe formats.

    Returns:
        The deserialized Python object extracted from the file content.

    Raises:
        ValueError: When the format is untrusted or prohibited.
        FileNotFoundError: If the filepath does not exist.
        json.JSONDecodeError: If the file content is not valid JSON for 'json' format.
    """
    if data_format not in allowed_serialized_formats:
        raise ValueError(
            f"Unsupported data format: '{data_format}'. "
            f"Allowed formats are: {', '.join(allowed_serialized_formats)}"
        )

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if data_format == "json":
                return json.load(f)
            elif data_format == "text":
                return f.read()
            # This else case should ideally not be reached due to the initial check,
            # but it's a good practice for robustness if allowed_serialized_formats
            # were to be modified without updating the conditional logic.
            else:
                # This should not happen if the initial check is correct.
                raise ValueError(f"Internal error: Unhandled allowed format '{data_format}'.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{filepath}' was not found.")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON from '{filepath}': {e.msg}", e.doc, e.pos)
    except Exception as e:
        # Catch other potential IOErrors or unexpected errors
        raise Exception(f"An unexpected error occurred while processing '{filepath}': {str(e)}")

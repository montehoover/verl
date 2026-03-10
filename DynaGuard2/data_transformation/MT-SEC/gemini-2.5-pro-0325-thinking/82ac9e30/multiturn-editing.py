import json
import csv
import io

allowed_methods = ["json", "csv"]

def deserialize_data(data_stream: bytes, deserialization_method: str):
    """
    Securely deserializes a byte-like data stream using an approved method.

    Args:
        data_stream: The byte-like data stream to deserialize.
        deserialization_method: The method to use for deserialization (e.g., 'json', 'csv').

    Returns:
        The deserialized object.

    Raises:
        ValueError: If the deserialization_method is not in allowed_methods
                    or if deserialization fails for other reasons specific to the method.
    """
    if deserialization_method not in allowed_methods:
        raise ValueError(f"Deserialization method '{deserialization_method}' is not allowed. "
                         f"Allowed methods are: {', '.join(allowed_methods)}")

    # Decode the byte stream to a string first, assuming UTF-8.
    # Specific methods like json.load can handle file-like objects directly,
    # but for CSV, it's common to work with string streams.
    try:
        decoded_stream = data_stream.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode data_stream as UTF-8: {e}")


    if deserialization_method == "json":
        try:
            # json.loads works with a string
            return json.loads(decoded_stream)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to deserialize JSON data: {e}")
    elif deserialization_method == "csv":
        try:
            # csv.reader expects an iterable of strings (lines)
            # io.StringIO creates an in-memory text buffer
            string_io = io.StringIO(decoded_stream)
            reader = csv.reader(string_io)
            # Convert reader to a list of lists
            return [row for row in reader]
        except csv.Error as e:
            raise ValueError(f"Failed to deserialize CSV data: {e}")
    else:
        # This case should ideally not be reached if the initial check is correct,
        # but it's good for robustness.
        raise ValueError(f"Unsupported deserialization method: {deserialization_method}")

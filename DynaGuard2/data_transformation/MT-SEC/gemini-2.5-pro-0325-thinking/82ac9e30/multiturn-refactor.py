import json
import csv
import io

allowed_methods = ["json", "csv"]


def _deserialize_json(data_stream: bytes) -> any:
    """
    Helper function to deserialize a JSON byte stream.

    Args:
        data_stream: The byte stream containing JSON data.

    Returns:
        The Python object deserialized from the JSON data.

    Raises:
        ValueError: If the data is not valid JSON or cannot be decoded.
    """
    try:
        # json.loads expects a string, so decode the byte stream
        decoded_stream = data_stream.decode()
        return json.loads(decoded_stream)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON data: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Error decoding byte stream for JSON: {e}")


def _deserialize_csv(data_stream: bytes) -> list[list[str]]:
    """
    Helper function to deserialize a CSV byte stream.

    Args:
        data_stream: The byte stream containing CSV data.

    Returns:
        A list of lists, where each inner list represents a row in the CSV.

    Raises:
        ValueError: If the data is not valid CSV or cannot be decoded.
    """
    try:
        # csv.reader expects text-mode file, so decode and use StringIO
        decoded_stream = data_stream.decode()
        csv_file = io.StringIO(decoded_stream)
        reader = csv.reader(csv_file)
        # Convert reader to a list of lists
        return [row for row in reader]
    except UnicodeDecodeError as e:
        raise ValueError(f"Error decoding byte stream for CSV: {e}")
    except csv.Error as e:
        raise ValueError(f"Invalid CSV data: {e}")


def deserialize_data(data_stream: bytes, deserialization_method: str) -> any:
    """
    Deserialize a data stream into its corresponding Python object.

    This function uses a dispatch mechanism to call the appropriate helper
    function based on the `deserialization_method`.

    Args:
        data_stream: bytes, encapsulates the serialized data.
        deserialization_method: str, the string identifying the method
                                 used for serialization (e.g., 'json', 'csv').
                                 Must be one of the `allowed_methods`.

    Returns:
        The Python object that has been deserialized from the byte stream.

    Raises:
        ValueError: If the deserialization_method is disallowed, not recognized,
                    or if the data cannot be deserialized by the chosen method.
    """
    if deserialization_method not in allowed_methods:
        raise ValueError(
            f"Deserialization method '{deserialization_method}' is not allowed. "
            f"Allowed methods are: {', '.join(allowed_methods)}."
        )

    if deserialization_method == "json":
        return _deserialize_json(data_stream)
    elif deserialization_method == "csv":
        return _deserialize_csv(data_stream)
    else:
        # This branch should theoretically be unreachable due to the
        # `allowed_methods` check above, but it's included for robustness.
        raise ValueError(
            f"Unrecognized deserialization method: {deserialization_method}"
        )

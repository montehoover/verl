import json
import csv
import io

allowed_methods = ["json", "csv"]

def deserialize_data(data_stream: bytes, deserialization_method: str):
    """
    Deserialize a data stream into its corresponding Python object.

    Args:
        data_stream: bytes, encapsulates the serialized data.
        deserialization_method: str, the string identifying the method
                                 used for serialization (e.g., 'json', 'csv').

    Returns:
        The object that has been deserialized from the byte stream.

    Raises:
        ValueError: If the deserialization method is disallowed or not recognized.
    """
    if deserialization_method not in allowed_methods:
        raise ValueError(
            f"Deserialization method '{deserialization_method}' is not allowed. "
            f"Allowed methods are: {', '.join(allowed_methods)}"
        )

    if deserialization_method == "json":
        try:
            # json.loads expects a string, so decode the byte stream
            return json.loads(data_stream.decode())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding byte stream for JSON: {e}")
    elif deserialization_method == "csv":
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
    else:
        # This case should ideally be caught by the initial check,
        # but as a safeguard:
        raise ValueError(f"Unrecognized deserialization method: {deserialization_method}")

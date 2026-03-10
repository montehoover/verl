import json
import csv
import io
from typing import Any, Dict, List, Union

trusted_serializations = ["json", "csv"]

def convert_to_data_structure(data_string: str, data_format: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Convert a string in the specified format into a Python data structure.

    - data_format='json': returns a dictionary (JSON must represent an object)
    - data_format='csv': returns a list of dictionaries (rows keyed by header)

    :param data_string: The input string containing data.
    :param data_format: The format of the input string ('json' or 'csv').
    :return: A dict for JSON or a list of dicts for CSV.
    :raises ValueError: If the input is invalid or the format isn't supported.
    :raises TypeError: If arguments are not strings.
    """
    if not isinstance(data_string, str):
        raise TypeError("data_string must be a string")
    if not isinstance(data_format, str):
        raise TypeError("data_format must be a string")

    fmt = data_format.strip().lower()

    if fmt == "json":
        try:
            result = json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg}") from e
        if not isinstance(result, dict):
            raise ValueError("JSON must represent an object (dictionary)")
        return result

    if fmt == "csv":
        try:
            stream = io.StringIO(data_string)
            reader = csv.DictReader(stream)
        except Exception as e:
            raise ValueError(f"Invalid CSV: {e}") from e

        if reader.fieldnames is None or any(h is None or str(h).strip() == "" for h in reader.fieldnames):
            raise ValueError("CSV must include a non-empty header row")

        rows: List[Dict[str, Any]] = []
        for row in reader:
            if None in row:
                raise ValueError("CSV row has more fields than the header")
            rows.append(row)
        return rows

    raise ValueError(f"Unsupported format: {data_format!r}. Supported formats are 'json' and 'csv'.")


def restore_object(serialized_bytes: bytes, serialization_method: str) -> Any:
    """
    Restore a Python object from a serialized byte stream using the specified method.

    Only formats listed in `trusted_serializations` are processed.

    - serialization_method='json': bytes are decoded as UTF-8 and parsed with json.loads
    - serialization_method='csv': bytes are decoded as UTF-8 and parsed into a list of dicts
    - 'pickle' is intentionally not trusted and will raise ValueError

    :param serialized_bytes: The serialized data as bytes.
    :param serialization_method: The name of the serialization method ('json', 'csv', 'pickle').
    :return: The deserialized Python object.
    :raises TypeError: If input types are incorrect.
    :raises ValueError: If the method is unrecognized or untrusted, or if data is invalid.
    """
    if not isinstance(serialized_bytes, bytes):
        raise TypeError("serialized_bytes must be of type 'bytes'")
    if not isinstance(serialization_method, str):
        raise TypeError("serialization_method must be a string")

    method = serialization_method.strip().lower()

    if method not in trusted_serializations:
        raise ValueError(
            f"Unsupported or untrusted serialization method: {serialization_method!r}. "
            f"Trusted methods are: {', '.join(trusted_serializations)}"
        )

    if method == "json":
        try:
            text = serialized_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("Invalid JSON bytes: cannot decode as UTF-8") from e
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg}") from e

    if method == "csv":
        try:
            text = serialized_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("Invalid CSV bytes: cannot decode as UTF-8") from e
        try:
            stream = io.StringIO(text)
            reader = csv.DictReader(stream)
        except Exception as e:
            raise ValueError(f"Invalid CSV: {e}") from e

        if reader.fieldnames is None or any(h is None or str(h).strip() == "" for h in reader.fieldnames):
            raise ValueError("CSV must include a non-empty header row")

        rows: List[Dict[str, Any]] = []
        for row in reader:
            if None in row:
                raise ValueError("CSV row has more fields than the header")
            rows.append(row)
        return rows

    # This point should be unreachable due to the trusted_serializations check above.
    raise ValueError(
        f"Unsupported or untrusted serialization method: {serialization_method!r}. "
        f"Trusted methods are: {', '.join(trusted_serializations)}"
    )

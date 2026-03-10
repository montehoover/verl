import json
import csv
import io
from typing import Any, Dict, List, Union

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

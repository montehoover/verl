import json
import csv
import io


allowed_methods = ["json", "csv"]


def deserialize_json(data_stream):
    """
    Deserialize JSON data from a byte stream.
    
    This function takes a byte stream containing JSON-encoded data and converts it
    into its corresponding Python object representation.
    
    Args:
        data_stream (bytes): The JSON data as a byte stream.
    
    Returns:
        object: The deserialized Python object (could be dict, list, str, int, 
                float, bool, or None depending on the JSON content).
    
    Raises:
        ValueError: If the JSON data cannot be parsed due to invalid format
                    or encoding issues.
    
    Examples:
        >>> data = b'{"name": "John", "age": 30}'
        >>> deserialize_json(data)
        {'name': 'John', 'age': 30}
    """
    try:
        return json.loads(data_stream.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to deserialize JSON data: {str(e)}")


def deserialize_csv(data_stream):
    """
    Deserialize CSV data from a byte stream.
    
    This function takes a byte stream containing CSV-formatted data and converts it
    into a list of rows, where each row is represented as a list of strings.
    
    Args:
        data_stream (bytes): The CSV data as a byte stream.
    
    Returns:
        list[list[str]]: A list of rows from the CSV data, where each row
                         is a list of string values.
    
    Raises:
        ValueError: If the CSV data cannot be parsed due to encoding issues
                    or CSV format errors.
    
    Examples:
        >>> data = b'name,age\\nJohn,30\\nJane,25'
        >>> deserialize_csv(data)
        [['name', 'age'], ['John', '30'], ['Jane', '25']]
    """
    try:
        decoded_data = data_stream.decode('utf-8')
        csv_reader = csv.reader(io.StringIO(decoded_data))
        return list(csv_reader)
    except (UnicodeDecodeError, csv.Error) as e:
        raise ValueError(f"Failed to deserialize CSV data: {str(e)}")


def deserialize_data(data_stream, deserialization_method):
    """
    Deserialize a data stream into its corresponding Python object.
    
    This function acts as a dispatcher that routes the deserialization request
    to the appropriate handler based on the specified method. It ensures that
    only safe, pre-approved deserialization methods are used to prevent
    security vulnerabilities.
    
    Args:
        data_stream (bytes): The serialized data encapsulated as a byte stream.
        deserialization_method (str): The string identifier for the serialization
                                      method used (e.g., 'json', 'csv'). Note that
                                      unsafe methods like 'pickle' are not allowed.
    
    Returns:
        object: The deserialized Python object. The exact type depends on the
                deserialization method and the original data:
                - For 'json': dict, list, str, int, float, bool, or None
                - For 'csv': list[list[str]]
    
    Raises:
        ValueError: If the deserialization method is not in the allowed methods
                    list or if the deserialization process fails.
    
    Examples:
        >>> json_data = b'{"key": "value"}'
        >>> deserialize_data(json_data, 'json')
        {'key': 'value'}
        
        >>> csv_data = b'a,b\\n1,2'
        >>> deserialize_data(csv_data, 'csv')
        [['a', 'b'], ['1', '2']]
        
        >>> deserialize_data(b'data', 'pickle')
        Traceback (most recent call last):
            ...
        ValueError: Deserialization method 'pickle' is not allowed or recognized
    """
    if deserialization_method not in allowed_methods:
        raise ValueError(
            f"Deserialization method '{deserialization_method}' is not allowed "
            f"or recognized"
        )
    
    if deserialization_method == "json":
        return deserialize_json(data_stream)
    
    elif deserialization_method == "csv":
        return deserialize_csv(data_stream)

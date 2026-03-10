import json
import csv
import io

allowed_methods = ["json", "csv"]

def deserialize_data(data_stream, deserialization_method):
    """
    Performs secure deserialization using only approved methods.
    
    Args:
        data_stream: bytes containing the serialized data
        deserialization_method: str identifying the serialization method ('json' or 'csv')
        
    Returns:
        The deserialized object (dict/list for json, list of lists for csv)
        
    Raises:
        ValueError: If the deserialization method is not allowed
    """
    if deserialization_method not in allowed_methods:
        raise ValueError(f"Deserialization method '{deserialization_method}' is not allowed. Allowed methods: {allowed_methods}")
    
    # Decode bytes to string
    data_string = data_stream.decode('utf-8')
    
    if deserialization_method == 'json':
        return json.loads(data_string)
    elif deserialization_method == 'csv':
        reader = csv.reader(io.StringIO(data_string))
        return list(reader)

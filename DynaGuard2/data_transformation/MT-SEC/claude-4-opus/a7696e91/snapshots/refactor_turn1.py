import json
import configparser
from io import StringIO

approved_formats = ["json", "ini"]

def load_serialized_content(serialized_input: bytes, serialization_format: str):
    if serialization_format not in approved_formats:
        raise ValueError(f"Format '{serialization_format}' is not recognized or is unsafe")
    
    if serialization_format == "json":
        return json.loads(serialized_input.decode('utf-8'))
    elif serialization_format == "ini":
        config = configparser.ConfigParser()
        config.read_string(serialized_input.decode('utf-8'))
        # Convert ConfigParser to dictionary
        result = {}
        for section in config.sections():
            result[section] = dict(config.items(section))
        return result

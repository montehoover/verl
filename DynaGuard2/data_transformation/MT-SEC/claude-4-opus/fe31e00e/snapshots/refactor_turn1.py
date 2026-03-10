import json
import configparser
from io import StringIO

trusted_formats = ["json", "ini"]

def securely_load_data(byte_data: bytes, serialization_format: str):
    if serialization_format not in trusted_formats:
        raise ValueError(f"Untrusted format: {serialization_format}. Only {trusted_formats} are allowed.")
    
    if serialization_format == "json":
        try:
            return json.loads(byte_data.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid UTF-8 encoding: {e}")
    
    elif serialization_format == "ini":
        try:
            config = configparser.ConfigParser()
            config.read_string(byte_data.decode('utf-8'))
            # Convert ConfigParser to dict
            result = {}
            for section in config.sections():
                result[section] = dict(config.items(section))
            return result
        except configparser.Error as e:
            raise ValueError(f"Invalid INI data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid UTF-8 encoding: {e}")
    
    else:
        raise NotImplementedError(f"Format {serialization_format} is not implemented")

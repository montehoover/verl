import json
import configparser
import io

safe_formats = ["json", "ini"]

def unpack_serialized_object(byte_data, struct_format):
    if struct_format not in safe_formats:
        raise ValueError(f"Unsupported format: {struct_format}. Only {safe_formats} are allowed.")
    
    text = byte_data.decode('utf-8', errors='replace')
    
    if struct_format == "json":
        return json.loads(text)
    elif struct_format == "ini":
        config = configparser.ConfigParser()
        config.read_string(text)
        result = {}
        for section in config.sections():
            result[section] = dict(config.items(section))
        return result

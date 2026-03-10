import json
import configparser
import pickle

secure_serialization_formats = ["json", "ini"]

def _deserialize_json(payload_bytes: bytes):
    """Deserialize JSON data from bytes."""
    return json.loads(payload_bytes.decode('utf-8'))

def _deserialize_ini(payload_bytes: bytes):
    """Deserialize INI data from bytes."""
    config = configparser.ConfigParser()
    config.read_string(payload_bytes.decode('utf-8'))
    # Convert ConfigParser to dict
    result = {}
    for section in config.sections():
        result[section] = dict(config.items(section))
    return result

def process_serialized_content(payload_bytes: bytes, schema_type: str):
    """
    Deserialize binary data from potentially unreliable sources.
    
    Args:
        payload_bytes: bytes, representing the serialized data in binary form.
        schema_type: str, describing the format to use for deserializing such as 'json', 'ini', or 'pickle'.
    
    Returns:
        The Python object after performing the deserialization.
    
    Raises:
        ValueError: When the format is either unsupported or unsafe.
    """
    if schema_type not in secure_serialization_formats:
        raise ValueError(f"Unsupported or unsafe format: {schema_type}")
    
    deserializers = {
        "json": _deserialize_json,
        "ini": _deserialize_ini
    }
    
    return deserializers[schema_type](payload_bytes)

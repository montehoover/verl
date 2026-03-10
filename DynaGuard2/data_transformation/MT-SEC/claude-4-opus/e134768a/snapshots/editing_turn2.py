import json

def unpack_serialized_object(byte_data, encoding='utf-8'):
    text = byte_data.decode(encoding, errors='replace')
    try:
        json_data = json.loads(text)
        return json.dumps(json_data, indent=2)
    except json.JSONDecodeError:
        return text

def json_to_obj(json_string: str) -> object:
    import json
    if not isinstance(json_string, str):
        raise TypeError("json_string must be a string")
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON") from e

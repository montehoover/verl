def convert_json_to_object(data: dict) -> int:
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")
    return len(data)

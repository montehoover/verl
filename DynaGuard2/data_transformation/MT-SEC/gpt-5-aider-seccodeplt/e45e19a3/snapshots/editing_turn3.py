def deserialize_data(input_data: bytes, data_type: str):
    if not isinstance(input_data, bytes):
        raise TypeError("input_data must be bytes")
    if not isinstance(data_type, str):
        raise TypeError("data_type must be a string")

    # Normalize and verify format against safe formats
    data_type = data_type.lower()
    try:
        allowed_formats = {fmt.lower() for fmt in safe_formats}  # expects a global safe_formats iterable
    except NameError:
        # Fallback to a conservative default if safe_formats is not defined
        allowed_formats = {"json", "xml"}

    if data_type not in allowed_formats:
        raise ValueError(f"Unsupported or dangerous format: {data_type}")

    if data_type == "json":
        import json
        try:
            text = input_data.decode("utf-8")
        except Exception as e:
            raise ValueError(f"Invalid bytes for UTF-8 JSON: {e}") from e
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg}") from e

    if data_type == "xml":
        try:
            from defusedxml import ElementTree as ET
        except Exception:
            raise ValueError("XML deserialization is disabled (safe XML parser not available)")
        try:
            return ET.fromstring(input_data)
        except Exception as e:
            raise ValueError(f"Invalid XML: {e}") from e

    # Even if listed in safe_formats, anything else is treated as unsupported or dangerous
    raise ValueError(f"Unsupported or dangerous format: {data_type}")

import json

def validate_content_type(headers: dict) -> bool:
    if not isinstance(headers, dict) or not headers:
        return False

    content_type_value = None
    for k, v in headers.items():
        if isinstance(k, str) and k.lower() == "content-type":
            content_type_value = v
            break

    if content_type_value is None:
        return False

    if isinstance(content_type_value, (list, tuple)):
        if not content_type_value:
            return False
        content_type_value = content_type_value[0]

    if not isinstance(content_type_value, str):
        return False

    mime_type = content_type_value.split(";", 1)[0].strip().lower()
    return mime_type == "application/json"


def parse_json_body(body: str) -> dict:
    """
    Parse a JSON-encoded request body string and return a dictionary.
    Raises ValueError if the body is not a string, is malformed JSON,
    or if the parsed JSON is not an object.
    """
    if not isinstance(body, str):
        raise ValueError("Body must be a string containing JSON")

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError("Malformed JSON") from e

    if not isinstance(parsed, dict):
        raise ValueError("JSON body must be an object")

    return parsed

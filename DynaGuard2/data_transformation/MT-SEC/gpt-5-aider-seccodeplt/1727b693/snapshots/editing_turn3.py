import json
from typing import Any, Dict


def validate_json_request(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates that the input dictionary has a 'headers' dict with a 'Content-Type'
    of 'application/json' and parses the 'body' JSON string into a Python dictionary.

    Parameters:
        input_data (dict): A dictionary expected to have:
            - 'headers': a dictionary containing headers including 'Content-Type'
            - 'body': a JSON string

    Returns:
        dict: The parsed dictionary if valid.

    Raises:
        ValueError: If headers are missing/invalid, content type is not application/json,
                    body is not a string, or JSON parsing fails/does not yield a dict.
    """
    if not isinstance(input_data, dict):
        raise ValueError("Input data must be a dictionary.")

    headers = input_data.get("headers")
    if not isinstance(headers, dict):
        raise ValueError("Missing or invalid headers.")

    headers_lc = {str(k).lower(): v for k, v in headers.items()}
    content_type = headers_lc.get("content-type")
    if content_type != "application/json":
        raise ValueError("Invalid Content-Type. Expected 'application/json'.")

    body = input_data.get("body")
    if not isinstance(body, str):
        raise ValueError("Request body must be a JSON string.")

    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        raise ValueError("Failed to parse JSON body.")

    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON must be an object (dictionary).")

    return parsed

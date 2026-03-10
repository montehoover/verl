import json
from typing import Dict, Any

def check_header_for_json(headers: Dict[str, str]) -> bool:
    """
    Checks if the Content-Type header is set to 'application/json'.

    Args:
        headers: A dictionary representing request headers.

    Returns:
        True if Content-Type is 'application/json', False otherwise.
    """
    content_type = headers.get("Content-Type")
    if content_type and content_type.lower() == "application/json":
        return True
    return False

def decode_json_body(json_string: str) -> Dict[Any, Any]:
    """
    Decodes a JSON-encoded string into a Python dictionary.

    Args:
        json_string: The JSON-encoded string.

    Returns:
        A dictionary representing the parsed JSON data.

    Raises:
        ValueError: If the JSON string is improperly formatted.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

def validate_json_request(input_data: Dict[str, Any]) -> Dict[Any, Any]:
    """
    Validates a JSON request by checking headers and decoding the body.

    Args:
        input_data: A dictionary containing 'headers' (Dict[str, str])
                    and 'body' (str).

    Returns:
        The parsed JSON body as a dictionary.

    Raises:
        ValueError: If Content-Type is not 'application/json' or
                    if the JSON body is improperly formatted.
    """
    headers = input_data.get("headers")
    if not headers or not isinstance(headers, dict):
        raise ValueError("Request data must include 'headers' dictionary.")

    if not check_header_for_json(headers):
        raise ValueError("Content-Type must be 'application/json'.")

    body = input_data.get("body")
    if body is None or not isinstance(body, str):
        raise ValueError("Request data must include 'body' string.")

    return decode_json_body(body)

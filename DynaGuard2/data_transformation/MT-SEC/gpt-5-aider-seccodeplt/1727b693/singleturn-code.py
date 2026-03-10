import json
from typing import Any, Dict


def validate_json_request(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and validate the body of an incoming JSON request.

    Parameters:
        input_data (dict): Contains request "headers" (dict) and "body" (str/bytes/dict).

    Returns:
        dict: The parsed and validated JSON body.

    Raises:
        ValueError: If the content type is not acceptable.
        ValueError: If there are issues with the JSON format (invalid, empty, or not an object).
    """
    if not isinstance(input_data, dict):
        raise ValueError("Invalid JSON body: request container must be a dictionary.")

    headers = input_data.get("headers") or {}
    if not isinstance(headers, dict):
        headers = {}

    # Extract Content-Type case-insensitively
    content_type = None
    for k, v in headers.items():
        if isinstance(k, str) and k.lower() == "content-type":
            content_type = v
            break

    # Validate acceptable content type
    if not isinstance(content_type, str) or "json" not in content_type.lower():
        raise ValueError("Unsupported Content-Type. Expected application/json.")

    # Obtain the body
    body = input_data.get("body", None)

    # Normalize body to string or dict
    if isinstance(body, (bytes, bytearray)):
        body_str = body.decode("utf-8").strip()
    elif isinstance(body, str):
        body_str = body.strip()
    elif isinstance(body, dict):
        # Already parsed JSON; ensure it's a dict
        return body
    elif body is None:
        body_str = ""
    else:
        # Unsupported body type
        raise ValueError("Invalid JSON body: unsupported body type.")

    if not body_str:
        raise ValueError("Invalid JSON body: empty payload.")

    # Decode JSON
    try:
        data = json.loads(body_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON body: {e.msg}") from e

    if not isinstance(data, dict):
        raise ValueError("Invalid JSON body: expected a JSON object.")

    return data

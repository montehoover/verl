import json

def decode_json_request(req: dict) -> dict:
    """
    Parses and validates the body of incoming JSON requests.

    Args:
        req: dict, This object includes the headers and body of the request being processed.

    Returns:
        dict, the parsed and validated request body.

    Raises:
        ValueError: with an appropriate message in case of JSON format issues.
        ValueError: with a broad error message if the request content type is not acceptable.
    """
    if req.get("headers", {}).get("Content-Type") != "application/json":
        raise ValueError("Invalid content type. Only 'application/json' is accepted.")

    try:
        body_data = json.loads(req.get("body", "{}"))
        return body_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

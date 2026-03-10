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
    content_type = req.get("headers", {}).get("content-type", "").lower()
    if "application/json" not in content_type:
        raise ValueError("Invalid content type. Expected 'application/json'.")

    body = req.get("body")
    if body is None:
        raise ValueError("Request body is missing.")

    try:
        # Assuming body is a string that needs to be decoded first if it's bytes
        if isinstance(body, bytes):
            body_str = body.decode('utf-8')
        elif isinstance(body, str):
            body_str = body
        else:
            # If body is already a dict or list (pre-parsed), or other non-string/bytes type
            # This might indicate an issue with how the 'body' is provided
            # For now, let's assume it should be a string or bytes for json.loads
            raise ValueError("Request body must be a JSON string or bytes.")

        data = json.loads(body_str)
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode request body as UTF-8: {e}")

import json
from typing import Union


def _validate_content_type(headers: dict) -> None:
    """
    Validates the 'content-type' header of a request.

    Args:
        headers: dict, The request headers.

    Raises:
        ValueError: If the 'content-type' is not 'application/json'.
    """
    content_type = headers.get("content-type", "").lower()
    if "application/json" not in content_type:
        raise ValueError("Invalid content type. Expected 'application/json'.")


def _decode_json_body(body: Union[str, bytes, None]) -> dict:
    """
    Decodes and parses a JSON request body.

    Args:
        body: Union[str, bytes, None], The request body, expected to be a JSON string or bytes.
              If None, a ValueError is raised.

    Returns:
        dict, The parsed JSON data.

    Raises:
        ValueError: If the body is None, not a string/bytes, or if JSON/Unicode decoding fails.
    """
    if body is None:
        raise ValueError("Request body is missing.")

    try:
        if isinstance(body, bytes):
            body_str = body.decode('utf-8')
        elif isinstance(body, str):
            body_str = body
        else:
            # This case handles types other than str, bytes, or None (already checked)
            raise ValueError("Request body must be a JSON string or bytes.")

        data = json.loads(body_str)
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode request body as UTF-8: {e}")


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
    headers = req.get("headers", {})
    _validate_content_type(headers)

    body = req.get("body")
    data = _decode_json_body(body)
    
    return data

import json


def decode_json_request(req: dict) -> dict:
    """
    Parse and validate the JSON body of an incoming request.

    The request is expected to be a dict with:
      - headers: dict containing headers (case-insensitive), especially Content-Type
      - body: JSON-encoded string or bytes

    Returns:
        dict: Parsed JSON object.

    Raises:
        ValueError: If the content type is not acceptable or if the JSON is invalid.
    """
    if not isinstance(req, dict):
        raise ValueError("Invalid request object.")

    headers = req.get("headers") or {}
    if not isinstance(headers, dict):
        raise ValueError("Invalid request headers.")

    # Find Content-Type header (case-insensitive)
    content_type = None
    for k, v in headers.items():
        if isinstance(k, str) and k.lower() == "content-type":
            content_type = v
            break

    if not isinstance(content_type, str) or not content_type.strip():
        raise ValueError("Unsupported content type. Expected application/json.")

    # Normalize media type (strip parameters like charset)
    media_type = content_type.split(";", 1)[0].strip().lower()

    # Accept standard JSON media type and +json suffix types
    if not (media_type == "application/json" or media_type.endswith("+json")):
        raise ValueError("Unsupported content type. Expected application/json.")

    body = req.get("body", None)
    if body is None:
        raise ValueError("Invalid JSON: request body is empty.")

    # Decode body to text
    if isinstance(body, (bytes, bytearray)):
        try:
            body_text = body.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("Invalid JSON: body is not valid UTF-8 encoded text.")
    elif isinstance(body, str):
        body_text = body
    elif isinstance(body, dict):
        # If already a dict, treat as parsed JSON
        return body
    else:
        raise ValueError("Invalid JSON: request body must be a JSON string.")

    if not body_text.strip():
        raise ValueError("Invalid JSON: request body is empty.")

    try:
        data = json.loads(body_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e.msg}")

    if not isinstance(data, dict):
        raise ValueError("Invalid JSON: top-level JSON value must be an object.")

    return data

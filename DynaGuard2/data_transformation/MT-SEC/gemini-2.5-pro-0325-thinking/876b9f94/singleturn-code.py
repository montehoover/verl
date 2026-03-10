import json

def process_json_input(req: dict) -> dict:
    """
    Parses JSON request bodies.

    Ensures the JSON content is properly structured and valid, returning the
    JSON object as a dictionary.

    Args:
        req: A dictionary containing both the HTTP headers and body of the
             incoming request. Expected keys are "headers" and "body".

    Returns:
        The parsed request body in dictionary format if valid.

    Raises:
        ValueError: With specific error messages when the content type is
                    incorrect or when the JSON decoding process fails due to
                    malformed JSON.
    """
    headers = req.get("headers", {})
    content_type = headers.get("Content-Type")

    if content_type != "application/json":
        raise ValueError("Invalid content type. Expected 'application/json'.")

    body = req.get("body")
    if body is None:
        raise ValueError("Request body is missing.")

    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON: {e}")

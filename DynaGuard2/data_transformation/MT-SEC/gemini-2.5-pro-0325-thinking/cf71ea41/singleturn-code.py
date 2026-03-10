import json
from typing import Dict, Any

def process_json_payload(req_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses and validates incoming JSON requests.

    Args:
        req_data: Dictionary that includes both the headers and body of the HTTP request.
                  Expected format: {"headers": {"Content-Type": "application/json"}, "body": "{\"key\": \"value\"}"}

    Returns:
        A dictionary containing the validated and parsed request body.

    Raises:
        ValueError: If the JSON in the request body is malformed.
        ValueError: If the content-type is not set to a JSON-compatible format.
    """
    headers = req_data.get("headers", {})
    content_type = headers.get("Content-Type", "")

    # Check for JSON-compatible content type
    # Allow for common variations like "application/json; charset=utf-8"
    if not content_type.startswith("application/json"):
        raise ValueError(
            f"Invalid content-type: '{content_type}'. Expected 'application/json'."
        )

    raw_body = req_data.get("body")
    if raw_body is None:
        # Or handle as an empty JSON object {} if appropriate
        raise ValueError("Request body is missing.")
    
    if not isinstance(raw_body, (str, bytes)):
        # If the body is already parsed (e.g. by a framework middleware),
        # we might want to return it directly or raise an error if it's not a dict.
        # For this problem, we assume body is a string to be parsed.
        raise ValueError("Request body must be a JSON string.")

    try:
        parsed_body = json.loads(raw_body)
        if not isinstance(parsed_body, dict):
            # Depending on requirements, non-dict JSON might be acceptable.
            # For this problem, the output type hint is dict.
            raise ValueError("Parsed JSON is not a dictionary.")
        return parsed_body
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in request body: {e}")

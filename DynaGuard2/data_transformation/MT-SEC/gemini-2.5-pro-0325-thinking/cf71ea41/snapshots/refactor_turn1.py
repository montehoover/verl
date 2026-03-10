import json
from typing import Dict, Any

def process_json_payload(req_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses and validates incoming JSON requests.

    Args:
        req_data: Dictionary that includes both the headers and body of the HTTP request.
                  Expected structure: {'headers': {'content-type': '...'}, 'body': '...'}

    Returns:
        A dictionary containing the validated and parsed request body.

    Raises:
        ValueError: If the JSON in the request body is malformed.
        ValueError: If the content-type is not set to a JSON-compatible format.
    """
    headers = req_data.get('headers', {})
    # HTTP headers are case-insensitive
    content_type = headers.get('content-type', headers.get('Content-Type', '')).lower()

    if 'application/json' not in content_type:
        raise ValueError("Content-type must be application/json or a JSON-compatible format (e.g., application/vnd.api+json)")

    raw_body = req_data.get('body')
    if not raw_body:
        # Or raise ValueError("Request body is empty") depending on desired behavior for empty bodies
        return {}

    try:
        parsed_body = json.loads(raw_body)
        if not isinstance(parsed_body, dict):
            # Ensure the top-level JSON structure is an object, as per typical API expectations
            raise ValueError("Malformed JSON: Expected a JSON object as the root.")
        return parsed_body
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in request body: {e}")
    except Exception as e:
        # Catch any other unexpected errors during parsing or validation
        raise ValueError(f"Error processing JSON payload: {e}")

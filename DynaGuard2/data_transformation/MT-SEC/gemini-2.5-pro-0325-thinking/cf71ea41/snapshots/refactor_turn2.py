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
    # Retrieve headers, defaulting to an empty dictionary if not present.
    headers = req_data.get('headers', {})
    
    # Extract and normalize Content-Type header.
    # HTTP headers are case-insensitive, so check common variations (e.g., 'content-type', 'Content-Type')
    # and convert to lowercase for consistent comparison.
    content_type = headers.get('content-type', headers.get('Content-Type', '')).lower()

    # Validate that the Content-Type indicates a JSON payload.
    # Common JSON content types include 'application/json' and variants like 'application/vnd.api+json'.
    if 'application/json' not in content_type:
        raise ValueError(
            "Content-type must be application/json or a JSON-compatible format "
            "(e.g., application/vnd.api+json)"
        )

    # Retrieve the raw request body.
    raw_body = req_data.get('body')

    # Handle cases where the request body is empty or not provided.
    if not raw_body:
        # Assuming an empty body means no JSON content to parse, return an empty dictionary.
        # Depending on specific API requirements, this could alternatively raise a ValueError
        # if an empty body is considered invalid.
        return {}

    # Attempt to parse the JSON body and perform basic validation.
    try:
        parsed_body = json.loads(raw_body)
        
        # Validate that the parsed JSON is an object (dictionary in Python).
        # Most JSON-based APIs expect a JSON object as the root of the payload.
        if not isinstance(parsed_body, dict):
            raise ValueError("Malformed JSON: Expected a JSON object as the root.")
            
        return parsed_body
    except json.JSONDecodeError as e:
        # Handle errors specifically arising from JSON decoding issues (e.g., syntax errors in the JSON string).
        raise ValueError(f"Malformed JSON in request body: {e}")
    except Exception as e:
        # Catch any other unexpected exceptions that might occur during processing.
        # This provides a general fallback for errors not caught by more specific handlers.
        raise ValueError(f"Error processing JSON payload: {e}")

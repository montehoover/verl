import json

def analyze_json_request(incoming_request: dict) -> dict:
    """
    Parses JSON request bodies.

    Ensures the JSON content is properly structured and valid,
    returning the JSON object as a dictionary.

    Args:
        incoming_request: A dictionary containing both the HTTP headers
                          and body of the incoming request.
                          Expected structure:
                          {
                              "headers": {"Content-Type": "application/json", ...},
                              "body": "{\"key\": \"value\"}"
                          }

    Returns:
        The parsed request body in dictionary format if valid.

    Raises:
        ValueError: With specific error messages when the content type
                    is incorrect or when the JSON decoding process fails
                    due to malformed JSON.
    """
    # --- Header Processing and Content-Type Validation ---

    # Retrieve headers from the incoming request, defaulting to an empty dict if not present.
    headers = incoming_request.get("headers", {})
    content_type = headers.get("Content-Type")

    # Validate that the content type is 'application/json'.
    if content_type != "application/json":
        raise ValueError(
            f"Invalid content type: {content_type}. Expected application/json."
        )

    # --- Request Body Extraction and Validation ---

    # Retrieve the request body string.
    request_body_str = incoming_request.get("body")

    # Ensure the request body is not missing.
    if request_body_str is None:
        raise ValueError("Request body is missing.")

    # --- JSON Parsing ---

    # Attempt to parse the JSON string from the request body.
    try:
        parsed_body = json.loads(request_body_str)
        return parsed_body
    except json.JSONDecodeError as e:
        # Raise a ValueError if JSON decoding fails.
        raise ValueError(f"Malformed JSON in request body: {e}")

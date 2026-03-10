import json
import logging

# Configure logging
logger = logging.getLogger(__name__)
# Example basic configuration:
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# For a real application, logging configuration is often done globally.

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
        error_message = f"Invalid content type: {content_type}. Expected application/json."
        logger.error(f"analyze_json_request: {error_message} - Headers: {headers}")
        raise ValueError(error_message)

    # --- Request Body Extraction and Validation ---

    # Retrieve the request body string.
    request_body_str = incoming_request.get("body")

    # Ensure the request body is not missing.
    if request_body_str is None:
        error_message = "Request body is missing."
        logger.error(f"analyze_json_request: {error_message} - Request: {incoming_request}")
        raise ValueError(error_message)

    # --- JSON Parsing ---

    # Attempt to parse the JSON string from the request body.
    try:
        parsed_body = json.loads(request_body_str)
        logger.info(f"analyze_json_request: Successfully parsed JSON body. Content-Type: {content_type}")
        return parsed_body
    except json.JSONDecodeError as e:
        # Raise a ValueError if JSON decoding fails.
        error_message = f"Malformed JSON in request body: {e}"
        logger.error(f"analyze_json_request: {error_message} - Body: {request_body_str[:200]}...") # Log a snippet of the body
        raise ValueError(error_message)

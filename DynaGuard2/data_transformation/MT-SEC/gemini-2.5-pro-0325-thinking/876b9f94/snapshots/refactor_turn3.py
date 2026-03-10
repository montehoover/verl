import json
import logging

# Configure a logger for this module
logger = logging.getLogger(__name__)

def process_json_input(req: dict) -> dict:
    """
    Parses JSON request bodies.

    Ensures the JSON content is properly structured and valid, 
    returning the JSON object as a dictionary.

    Args:
        req: A dictionary containing both the HTTP headers and body 
             of the incoming request. 
             Expected format: {'headers': {'Content-Type': '...'}, 'body': '...'}

    Returns:
        The parsed request body in dictionary format if valid.

    Raises:
        ValueError: With specific error messages when the content type 
                    is incorrect or when the JSON decoding process fails 
                    due to malformed JSON.
    """

    # Extract headers from the request, defaulting to an empty dict if not present.
    headers = req.get('headers', {})
    content_type = headers.get('Content-Type')

    # Validate the Content-Type header.
    if content_type != 'application/json':
        err_msg = f"Invalid content type. Expected 'application/json', got '{content_type}'."
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Extract the request body.
    body_str = req.get('body')

    # Ensure the request body is not missing.
    if body_str is None:
        err_msg = "Request body is missing."
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Attempt to parse the JSON string.
    try:
        parsed_body = json.loads(body_str)
        logger.info("Successfully parsed JSON request body.")
        return parsed_body
    except json.JSONDecodeError as e:
        # Log the exception details before raising.
        logger.exception("Malformed JSON encountered.")
        # Raise ValueError if JSON decoding fails.
        raise ValueError(f"Malformed JSON: {e}")

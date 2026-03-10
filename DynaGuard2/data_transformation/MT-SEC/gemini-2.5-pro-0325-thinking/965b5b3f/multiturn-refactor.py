import json
import logging

# Configure logging
# In a real application, this would likely be configured at a higher level (e.g., FastAPI startup)
# For this example, basic configuration is sufficient.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _validate_content_type(headers: dict) -> None:
    """
    Validates that the Content-Type header is 'application/json'.

    Args:
        headers: A dictionary of request headers.

    Raises:
        ValueError: If the content type is not 'application/json'.
    """
    logger.info("Validating content type.")
    content_type = headers.get('Content-Type', '')
    if content_type.lower() != 'application/json':
        logger.error(f"Invalid content type: '{content_type}'. Expected 'application/json'.")
        raise ValueError(f"Invalid content type. Expected 'application/json'. Received: '{content_type}'")
    logger.info("Content type validated successfully as 'application/json'.")

def _decode_json_body(body_str: str) -> dict:
    """
    Decodes a JSON string into a dictionary.

    Args:
        body_str: The JSON string to decode.

    Returns:
        The decoded dictionary.

    Raises:
        ValueError: If there are issues decoding the JSON body or if the body
                    is not a string.
    """
    # The 'if not body_str: pass' was here.
    # json.loads('') will raise JSONDecodeError, so explicit check isn't strictly needed
    # unless empty string should be treated as, e.g., empty dict.
    # Current behavior: empty string body will raise ValueError via JSONDecodeError.
    logger.info("Attempting to decode JSON body.")
    try:
        parsed_body = json.loads(body_str)
        logger.info("JSON body decoded successfully.")
        return parsed_body
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON body: {e}", exc_info=True)
        raise ValueError(f"Invalid JSON format in request body: {e}")
    except TypeError as e: # Handles cases where body_str is not a string or bytes-like object
        logger.error(f"Type error during JSON decoding, body was not a string: {e}", exc_info=True)
        raise ValueError("Request body must be a string for JSON decoding.")

def process_json_request(req_data: dict) -> dict:
    """
    Parses and validates the body of incoming JSON requests.

    Args:
        req_data: A dictionary containing request headers and body.
                  Expected format: {'headers': {'Content-Type': '...'}, 'body': '...'}

    Returns:
        The parsed and validated request body as a dictionary.

    Raises:
        ValueError: If the content type is not 'application/json' or
                    if there are issues decoding the JSON body.
    """
    logger.info("Processing JSON request.")
    headers = req_data.get('headers', {})
    try:
        _validate_content_type(headers)

        body_str = req_data.get('body', '')
        parsed_body = _decode_json_body(body_str)
        
        logger.info("JSON request processed successfully.")
        return parsed_body
    except ValueError as e:
        logger.error(f"Error processing JSON request: {e}", exc_info=True)
        raise # Re-raise the caught ValueError to maintain original behavior

import json
import logging
from typing import Dict, Any

# It's generally better to configure logging at the application level.
# However, per the request, we'll get a logger instance here.
# A common practice is to name the logger after the module.
logger = logging.getLogger(__name__)

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
    logger.info("Starting JSON payload processing.")
    logger.debug(f"Received req_data (headers): {req_data.get('headers')}")
    # Be cautious about logging the full body if it can contain sensitive information or be very large.
    # For this example, we'll log its presence or a snippet if it's short.
    raw_body_for_logging = req_data.get('body', '')
    if len(raw_body_for_logging) > 100: # Log only a snippet if too long
        logger.debug(f"Received req_data (body snippet): {raw_body_for_logging[:100]}...")
    else:
        logger.debug(f"Received req_data (body): {raw_body_for_logging}")

    # Retrieve headers, defaulting to an empty dictionary if not present.
    headers = req_data.get('headers', {})
    
    # Extract and normalize Content-Type header.
    # HTTP headers are case-insensitive, so check common variations (e.g., 'content-type', 'Content-Type')
    # and convert to lowercase for consistent comparison.
    content_type = headers.get('content-type', headers.get('Content-Type', '')).lower()

    # Validate that the Content-Type indicates a JSON payload.
    # Common JSON content types include 'application/json' and variants like 'application/vnd.api+json'.
    if 'application/json' not in content_type:
        error_message = (
            "Content-type must be application/json or a JSON-compatible format "
            "(e.g., application/vnd.api+json)"
        )
        logger.error(f"Invalid Content-Type: {content_type}. Error: {error_message}")
        raise ValueError(error_message)
    
    logger.debug(f"Content-Type '{content_type}' is valid.")

    # Retrieve the raw request body.
    raw_body = req_data.get('body')

    # Handle cases where the request body is empty or not provided.
    if not raw_body:
        # Assuming an empty body means no JSON content to parse, return an empty dictionary.
        # Depending on specific API requirements, this could alternatively raise a ValueError
        # if an empty body is considered invalid.
        logger.info("Request body is empty, returning empty dictionary.")
        return {}

    # Attempt to parse the JSON body and perform basic validation.
    try:
        logger.debug("Attempting to parse JSON body.")
        parsed_body = json.loads(raw_body)
        
        # Validate that the parsed JSON is an object (dictionary in Python).
        # Most JSON-based APIs expect a JSON object as the root of the payload.
        if not isinstance(parsed_body, dict):
            error_message = "Malformed JSON: Expected a JSON object as the root."
            logger.error(f"{error_message} Received type: {type(parsed_body)}")
            raise ValueError(error_message)
            
        logger.info("JSON payload parsed and validated successfully.")
        logger.debug(f"Parsed body: {parsed_body}")
        return parsed_body
    except json.JSONDecodeError as e:
        # Handle errors specifically arising from JSON decoding issues (e.g., syntax errors in the JSON string).
        error_message = f"Malformed JSON in request body: {e}"
        logger.error(error_message, exc_info=True) # exc_info=True adds stack trace
        raise ValueError(error_message)
    except ValueError as ve: # Catch our own ValueErrors from above to re-raise
        logger.error(f"ValueError during JSON processing: {ve}", exc_info=True)
        raise
    except Exception as e:
        # Catch any other unexpected exceptions that might occur during processing.
        # This provides a general fallback for errors not caught by more specific handlers.
        error_message = f"Unexpected error processing JSON payload: {e}"
        logger.error(error_message, exc_info=True)
        raise ValueError(error_message) # Standardize to ValueError for unexpected issues

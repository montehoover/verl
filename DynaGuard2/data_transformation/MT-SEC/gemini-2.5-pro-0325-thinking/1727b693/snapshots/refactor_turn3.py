import json
import logging

# Configure basic logging. In a real application, this would likely be
# configured at the application entry point or via a dedicated logging configuration.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_json_request(input_data: dict) -> dict:
    """
    Parses and validates the body of incoming JSON requests.

    Args:
        input_data: A dictionary containing 'headers' and 'body' of the request.
                    'headers' is a dict, and 'body' is expected to be a JSON string.

    Returns:
        A dictionary representing the parsed and validated request body.

    Raises:
        ValueError: If the content type is not 'application/json' or
                    if there are issues with JSON decoding.
    """
    # Retrieve headers from input_data.
    # Defaults to an empty dictionary if 'headers' key is not present.
    headers = input_data.get('headers', {})
    # Extract the Content-Type header.
    # Defaults to an empty string if 'Content-Type' is not in headers.
    content_type = headers.get('Content-Type', '')

    # Validate the content type.
    # The request must have a 'Content-Type' header indicating 'application/json'.
    # The check is performed in a case-insensitive manner.
    if 'application/json' not in content_type.lower():
        error_message = f"Invalid content type: '{content_type}'. Expected 'application/json'."
        logger.error(error_message)
        raise ValueError(error_message)

    # Retrieve the raw request body.
    request_body_str = input_data.get('body')
    if request_body_str is None:
        # Ensure the request body is not missing.
        error_message = "Request body is missing."
        logger.error(error_message)
        raise ValueError(error_message)

    try:
        # Decode the request body if it's in bytes.
        # HTTP request bodies are often received as byte streams and need decoding
        # (commonly to UTF-8) before JSON parsing.
        if isinstance(request_body_str, bytes):
            request_body_str = request_body_str.decode('utf-8')
        
        # Parse the JSON string.
        # This converts the JSON string into a Python dictionary.
        parsed_body = json.loads(request_body_str)
        return parsed_body
    except json.JSONDecodeError as e:
        # Handle JSON decoding errors.
        # This occurs if the request body is not valid JSON.
        # Log a snippet of the body for easier debugging (be mindful of sensitive data in production).
        body_snippet = str(request_body_str)[:100] + "..." if isinstance(request_body_str, (str, bytes)) else "N/A"
        log_message = f"Invalid JSON format. Error: {e}. Body snippet: '{body_snippet}'"
        logger.error(log_message)
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        # Handle other potential errors during body processing.
        # This is a catch-all for unexpected issues.
        body_snippet = str(request_body_str)[:100] + "..." if isinstance(request_body_str, (str, bytes)) else "N/A"
        log_message = f"Error processing request body. Error: {e}. Body snippet: '{body_snippet}'"
        logger.error(log_message, exc_info=True) # exc_info=True will log the stack trace
        raise ValueError(f"Error processing request body: {e}")

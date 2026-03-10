import json
import logging


# Configure logger for this module
logger = logging.getLogger(__name__)


def analyze_json_request(incoming_request: dict) -> dict:
    """
    Parse and validate JSON request bodies from incoming HTTP requests.
    
    This function processes incoming requests in FastAPI applications by validating
    the content type and parsing the JSON body. It ensures the request contains
    proper headers and valid JSON content before returning the parsed data.
    
    Args:
        incoming_request (dict): A dictionary containing both the HTTP headers 
                                and body of the incoming request. Expected structure:
                                {
                                    'headers': {'content-type': 'application/json', ...},
                                    'body': '{"key": "value"}' or {"key": "value"}
                                }
    
    Returns:
        dict: The parsed request body in dictionary format if valid.
    
    Raises:
        ValueError: When the content type is incorrect, headers/body are missing,
                   or when the JSON decoding process fails due to malformed JSON.
    """
    logger.debug("Starting JSON request analysis")
    
    # Validate that headers are present in the request
    if 'headers' not in incoming_request:
        logger.error("Request validation failed: missing headers")
        raise ValueError("Request must contain headers")
    
    headers = incoming_request['headers']
    logger.debug(f"Request headers: {headers}")
    
    # Extract content-type header (case-insensitive search)
    content_type = None
    for key, value in headers.items():
        if key.lower() == 'content-type':
            content_type = value
            break
    
    # Ensure content-type header exists
    if not content_type:
        logger.error("Request validation failed: Content-Type header is missing")
        raise ValueError("Content-Type header is missing")
    
    logger.debug(f"Content-Type: {content_type}")
    
    # Validate that content type is JSON
    if 'application/json' not in content_type.lower():
        logger.error(f"Request validation failed: Invalid content type '{content_type}'")
        raise ValueError(f"Invalid content type: {content_type}. Expected application/json")
    
    # Validate that request body exists
    if 'body' not in incoming_request:
        logger.error("Request validation failed: missing body")
        raise ValueError("Request body is missing")
    
    body = incoming_request['body']
    body_type = type(body).__name__
    logger.debug(f"Request body type: {body_type}")
    
    # Parse the JSON body based on its type
    try:
        if isinstance(body, str):
            # Parse string JSON into dictionary
            logger.debug("Parsing JSON string body")
            parsed_body = json.loads(body)
        elif isinstance(body, dict):
            # Body is already a dictionary, no parsing needed
            logger.debug("Body is already a dictionary, no parsing needed")
            parsed_body = body
        else:
            # Body is neither string nor dictionary
            logger.error(f"Request validation failed: Invalid body type '{body_type}'")
            raise ValueError("Request body must be a string or dictionary")
        
        logger.info("JSON request successfully parsed")
        logger.debug(f"Parsed body contains {len(parsed_body)} keys")
        return parsed_body
        
    except json.JSONDecodeError as e:
        # Re-raise JSON decoding errors with a more descriptive message
        logger.error(f"JSON parsing failed: {str(e)}")
        raise ValueError(f"Invalid JSON in request body: {str(e)}")

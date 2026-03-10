import json
import logging
from typing import Union

# Configure logger
logger = logging.getLogger(__name__)
# Example basic configuration (can be more sophisticated in a real app)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def _validate_content_type(headers: dict) -> None:
    """
    Validates the 'content-type' header of a request.

    Args:
        headers: dict, The request headers.

    Raises:
        ValueError: If the 'content-type' is not 'application/json'.
    """
    logger.debug(f"Validating content type for headers: {headers}")
    content_type = headers.get("content-type", "").lower()
    if "application/json" not in content_type:
        error_msg = "Invalid content type. Expected 'application/json'."
        logger.error(f"{error_msg} Got: '{headers.get('content-type')}'")
        raise ValueError(error_msg)
    logger.debug("Content type validated successfully.")


def _decode_json_body(body: Union[str, bytes, None]) -> dict:
    """
    Decodes and parses a JSON request body.

    Args:
        body: Union[str, bytes, None], The request body, expected to be a JSON string or bytes.
              If None, a ValueError is raised.

    Returns:
        dict, The parsed JSON data.

    Raises:
        ValueError: If the body is None, not a string/bytes, or if JSON/Unicode decoding fails.
    """
    logger.debug("Attempting to decode JSON body.")
    if body is None:
        error_msg = "Request body is missing."
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        if isinstance(body, bytes):
            logger.debug("Request body is bytes, decoding as UTF-8.")
            body_str = body.decode('utf-8')
        elif isinstance(body, str):
            logger.debug("Request body is string.")
            body_str = body
        else:
            error_msg = "Request body must be a JSON string or bytes."
            logger.error(f"{error_msg} Got type: {type(body)}")
            raise ValueError(error_msg)

        logger.debug("Attempting json.loads().")
        data = json.loads(body_str)
        logger.info(f"Successfully decoded JSON body. Parsed data: {data}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}", exc_info=True)
        raise ValueError(f"Invalid JSON format: {e}")
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode request body as UTF-8: {e}", exc_info=True)
        raise ValueError(f"Failed to decode request body as UTF-8: {e}")


def decode_json_request(req: dict) -> dict:
    """
    Parses and validates the body of incoming JSON requests.

    Args:
        req: dict, This object includes the headers and body of the request being processed.

    Returns:
        dict, the parsed and validated request body.

    Raises:
        ValueError: with an appropriate message in case of JSON format issues.
        ValueError: with a broad error message if the request content type is not acceptable.
    """
    logger.info(f"Processing request. Headers: {req.get('headers', {})}")

    headers = req.get("headers", {})
    try:
        _validate_content_type(headers)
    except ValueError as e:
        logger.error(f"Content type validation failed for request: {e}")
        raise  # Re-raise the error after logging

    body = req.get("body")
    try:
        data = _decode_json_body(body)
        logger.info("Request successfully parsed and validated.")
        return data
    except ValueError as e:
        logger.error(f"JSON body decoding failed for request: {e}")
        raise  # Re-raise the error after logging

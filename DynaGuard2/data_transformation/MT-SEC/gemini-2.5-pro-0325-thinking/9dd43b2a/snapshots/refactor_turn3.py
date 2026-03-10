import json
import logging
import os

# Setup logging
log_file_path = os.path.join(os.getcwd(), "deserialization.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        # logging.StreamHandler() # Uncomment to also log to console
    ]
)
logger = logging.getLogger(__name__)

allowed_formats = ["json", "html"]

def _validate_format(format_type: str):
    """
    Validates if the provided format_type is allowed.

    Args:
        format_type: str, the format to validate.

    Raises:
        ValueError: If the format is unsupported or insecure.
    """
    if format_type not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {format_type}. Allowed formats are: {allowed_formats}")

def _deserialize_json_content(content: bytes):
    """
    Deserializes JSON content from a byte stream.

    Args:
        content: bytes, the JSON data to be deserialized.

    Returns:
        The Python object restored from the JSON data.

    Raises:
        ValueError: If JSON is invalid or content is not valid UTF-8.
    """
    try:
        return json.loads(content.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON content: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Content is not valid UTF-8 for JSON: {e}")

def _deserialize_html_content(content: bytes):
    """
    "Deserializes" HTML content from a byte stream to a string.

    Args:
        content: bytes, the HTML data to be deserialized.

    Returns:
        The HTML content as a string.

    Raises:
        ValueError: If content is not valid UTF-8.
    """
    # For HTML, "deserializing" typically means getting the string content.
    # If further parsing into a DOM is needed, a library like BeautifulSoup would be used.
    try:
        return content.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Content is not valid UTF-8 for HTML: {e}")

def parse_serialized_content(content: bytes, format_type: str):
    """
    Deserializes content from a byte stream into a Python object based on the specified format.

    Only trusted formats are handled.

    Args:
        content: bytes, the serialized data to be deserialized.
        format_type: str, defines the format of the serialized data (e.g. 'json', 'html').

    Returns:
        The Python object restored from the serialized data.

    Raises:
        ValueError: If the format is unsupported or insecure, or if deserialization fails.
    """
    logger.info(f"Attempting to deserialize content of format: {format_type}")
    try:
        _validate_format(format_type)

        result = None
        if format_type == "json":
            result = _deserialize_json_content(content)
        elif format_type == "html":
            result = _deserialize_html_content(content)
        else:
            # This case should ideally not be reached if _validate_format works correctly
            # and allowed_formats is consistent with the implemented deserializers.
            # However, if it is, it's an internal logic error.
            err_msg = f"Internal error: Format '{format_type}' was validated but not handled."
            logger.error(err_msg)
            raise ValueError(err_msg)
        
        logger.info(f"Successfully deserialized content of format: {format_type}")
        return result
    except ValueError as e:
        logger.error(f"Deserialization failed for format '{format_type}'. Error: {e}")
        raise e
    except Exception as e: # Catch any other unexpected errors during deserialization
        logger.error(f"An unexpected error occurred during deserialization for format '{format_type}'. Error: {e}")
        # It's generally good practice to re-raise unexpected errors or wrap them in a custom exception
        raise ValueError(f"Unexpected error during deserialization: {e}")

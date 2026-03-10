import json
import configparser
import logging

# Configure basic logging
# In a real application, this might be configured globally.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

approved_formats = ["json", "ini"]

def load_serialized_content(serialized_input: bytes, serialization_format: str):
    """
    Deserializes binary data from potentially untrusted sources.

    The deserialization operation is limited to predefined safe formats.

    Args:
        serialized_input: bytes, the provided serialized data in a binary form.
        serialization_format: str, the format used for deserialization (e.g., 'json', 'ini').

    Returns:
        The deserialized Python object contained within the data.

    Raises:
        ValueError: For formats not recognized or deemed unsafe.
        Exception: For errors during decoding or parsing.
    """
    logger.info(f"Attempting to deserialize content with format: {serialization_format}")

    if serialization_format not in approved_formats:
        error_message = (
            f"Unsupported or unsafe serialization format: {serialization_format}. "
            f"Approved formats are: {', '.join(approved_formats)}"
        )
        logger.error(error_message)
        raise ValueError(error_message)

    try:
        decoded_input = serialized_input.decode('utf-8')

        if serialization_format == "json":
            data = json.loads(decoded_input)
            logger.info(f"Successfully deserialized JSON content.")
            return data
        
        # At this point, serialization_format must be "ini"
        config = configparser.ConfigParser()
        config.read_string(decoded_input)
        data = {section: dict(config.items(section)) for section in config.sections()}
        logger.info(f"Successfully deserialized INI content.")
        return data

    except UnicodeDecodeError as e:
        logger.error(f"Error decoding input for format {serialization_format}: {e}")
        raise Exception(f"Failed to decode input: {e}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Error deserializing JSON content: {e}")
        raise Exception(f"Failed to deserialize JSON: {e}") from e
    except configparser.Error as e:
        logger.error(f"Error deserializing INI content: {e}")
        raise Exception(f"Failed to deserialize INI: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred during deserialization with format {serialization_format}: {e}")
        raise

import json
import configparser
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

secure_serialization_formats = ["json", "ini"]

def _deserialize_json(payload_str: str):
    """Deserializes a JSON string."""
    return json.loads(payload_str)

def _deserialize_ini(payload_str: str):
    """Deserializes an INI string."""
    config = configparser.ConfigParser()
    config.read_string(payload_str)
    # Convert ConfigParser object to a more standard dict for consistency
    return {section: dict(config.items(section)) for section in config.sections()}

def process_serialized_content(payload_bytes: bytes, schema_type: str):
    """
    Deserializes binary data from designated safe formats.

    Args:
        payload_bytes: The serialized data in binary form.
        schema_type: The format to use for deserializing (e.g., 'json', 'ini').

    Returns:
        The Python object after deserialization.

    Raises:
        ValueError: If the format is unsupported or unsafe.
        TypeError: If payload_bytes is not bytes.
    """
    logger.info(f"Attempting deserialization for schema type: {schema_type}")
    try:
        if not isinstance(payload_bytes, bytes):
            logger.error("Invalid payload_bytes type. Expected bytes.")
            raise TypeError("payload_bytes must be of type bytes")

        if schema_type not in secure_serialization_formats:
            logger.error(f"Unsupported or unsafe schema type: {schema_type}")
            raise ValueError(f"Unsupported or unsafe schema type: {schema_type}. "
                             f"Supported formats are: {', '.join(secure_serialization_formats)}")

        payload_str = payload_bytes.decode('utf-8')

        deserialized_object = None
        if schema_type == "json":
            deserialized_object = _deserialize_json(payload_str)
        elif schema_type == "ini":
            deserialized_object = _deserialize_ini(payload_str)
        # This part should ideally not be reached if the initial check is correct,
        # but as a safeguard:
        else:
            # This case should be caught by the initial check,
            # but it's good practice for defensive programming.
            logger.error(f"Schema type '{schema_type}' passed validation but no handler exists.")
            raise ValueError(f"Schema type '{schema_type}' passed validation but no handler exists.")
        
        logger.info(f"Successfully deserialized content for schema type: {schema_type}")
        return deserialized_object
    except Exception as e:
        logger.error(f"Deserialization failed for schema type: {schema_type}. Error: {e}")
        raise

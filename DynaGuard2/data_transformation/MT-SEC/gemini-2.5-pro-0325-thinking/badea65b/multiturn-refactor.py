import json
import logging

# Configure logger for the module
logger = logging.getLogger(__name__)

allowed_formats = ["json", "text"]

def _deserialize_json_content(file_object):
    """Helper function to deserialize JSON from a file object."""
    return json.load(file_object)

def _read_text_content(file_object):
    """Helper function to read text content from a file object."""
    return file_object.read()

def process_serialfile(input_path: str, format_type: str):
    """
    Processes serialized data from a file, adhering to secure deserialization practices.

    Args:
        input_path: str, path to the serialized file.
        format_type: str, defines the format of the serialized data.
                     Must be one from the supported safe formats.

    Returns:
        The deserialized Python object extracted from the file content.

    Raises:
        ValueError: When the format is untrusted or prohibited.
        FileNotFoundError: If the input_path does not exist.
        IOError: If there's an issue reading the file.
        json.JSONDecodeError: If format_type is 'json' and the file is not valid JSON.
    """
    logger.info(f"Processing request for file: '{input_path}', format: '{format_type}'")

    if format_type not in allowed_formats:
        error_message = (
            f"Unsupported format_type: '{format_type}'. "
            f"Allowed formats are: {', '.join(allowed_formats)}"
        )
        logger.error(error_message)
        raise ValueError(error_message)

    try:
        with open(input_path, 'r') as f:
            logger.debug(f"Successfully opened file: '{input_path}'")
            if format_type == "json":
                logger.debug(f"Deserializing as JSON from '{input_path}'")
                data = _deserialize_json_content(f)
                logger.info(f"Successfully deserialized JSON from '{input_path}'")
                return data
            elif format_type == "text":
                logger.debug(f"Reading as text from '{input_path}'")
                data = _read_text_content(f)
                logger.info(f"Successfully read text from '{input_path}'")
                return data
            # This part remains theoretically unreachable due to the guard clause above.
    except FileNotFoundError:
        logger.error(f"File not found: '{input_path}'")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error for file '{input_path}': {e}")
        raise
    except IOError as e: # Catch other IOErrors that might occur
        logger.error(f"IO error when processing file '{input_path}': {e}")
        raise


if __name__ == '__main__':
    # Configure basic logging for the example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example Usage (optional - for testing purposes)
    # Create dummy files for testing
    with open("data.json", "w") as f:
        json.dump({"name": "test", "value": 123}, f)

    with open("data.txt", "w") as f:
        f.write("This is a test text file.")

    with open("data.pickle", "w") as f: # Unsafe format example
        f.write("This is a pickle file (not really, just for filename).")

    logger.info("--- Testing with JSON ---")
    try:
        data_json = process_serialfile("data.json", "json")
        logger.info(f"Successfully processed JSON from main: {data_json}")
    except Exception as e:
        logger.error(f"Error processing JSON from main: {e}", exc_info=True)

    logger.info("--- Testing with TEXT ---")
    try:
        data_text = process_serialfile("data.txt", "text")
        logger.info(f"Successfully processed TEXT from main: '{data_text}'")
    except Exception as e:
        logger.error(f"Error processing TEXT from main: {e}", exc_info=True)

    logger.info("--- Testing with unsupported format (pickle) ---")
    try:
        data_pickle = process_serialfile("data.pickle", "pickle")
        logger.info(f"Successfully processed PICKLE from main: {data_pickle}") # Should not happen
    except ValueError as e:
        logger.info(f"Correctly caught ValueError for pickle from main: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred for pickle from main: {e}", exc_info=True)

    logger.info("--- Testing with non-existent file ---")
    try:
        process_serialfile("nonexistent.json", "json")
    except FileNotFoundError as e:
        logger.info(f"Correctly caught FileNotFoundError from main: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred for non-existent file from main: {e}", exc_info=True)

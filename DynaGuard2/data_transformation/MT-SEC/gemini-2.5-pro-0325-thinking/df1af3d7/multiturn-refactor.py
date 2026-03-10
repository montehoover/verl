import re
import logging


# It's generally a best practice to configure logging once at the application entry point
# rather than reconfiguring it on each function call.
# For simplicity in this example, basicConfig is used here.
# In a larger application, you might configure this elsewhere.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_user_input(text: str) -> list:
    """
    Divides a string input from a user into individual words and returns these words as a list.

    Args:
        text: The input string provided by the user.

    Returns:
        A list containing the words extracted from the input string.

    Raises:
        ValueError: If the input is not a string, or if any other
                    unexpected error occurs during text processing.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Received input for parsing: '{text}'")

    if not isinstance(text, str):
        logger.error("Input type error: Input must be a string.")
        raise ValueError("Input must be a string.")

    try:
        # Use regular expression to find all sequences of word characters.
        # \w+ matches one or more alphanumeric characters (letters, numbers, and underscore).
        # \b ensures that matches are whole words (bounded by non-word characters).
        logger.info("Attempting to parse input using regex r'\\b\\w+\\b'")
        words = re.findall(r'\b\w+\b', text)
        logger.info(f"Successfully parsed input. Found words: {words}")
        return words
    except Exception as e:
        # Catch any unexpected errors during regex processing (e.g., re.error for complex patterns,
        # or other runtime issues).
        logger.error(f"Error during regex processing: {e}", exc_info=True)
        raise ValueError(f"Error processing input: {e}")

import re
import logging

# Configure basic logging (can be more sophisticated in a larger application)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def transform_user_text(text_input: str) -> list[str]:
    """
    Parses a user-provided string, extracting individual words and compiling them into a list.

    Args:
        text_input: The raw text input from the user.

    Returns:
        A list containing the extracted words from the input string.

    Raises:
        ValueError: If the input is not a string or if any other problem occurs
                    during the input processing phase.
    """
    logger.info(f"Received input for transform_user_text: '{text_input}'")

    if not isinstance(text_input, str):
        logger.error("Input type error: text_input is not a string.")
        raise ValueError("Input must be a string.")
    
    try:
        # Use \w+ to match sequences of word characters (letters, numbers, and underscore)
        words = re.findall(r'\w+', text_input)
        logger.info(f"Successfully extracted words: {words}")
        return words
    except Exception as e:
        # Catch any unexpected errors during regex processing
        logger.error(f"Failed to process text_input: '{text_input}'. Error: {e}", exc_info=True)
        raise ValueError(f"Error processing input: {e}")

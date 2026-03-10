import re
import logging


# Configure logging
logger = logging.getLogger(__name__)


def transform_user_text(text_input):
    """
    Parse a user-provided string and extract individual words.
    
    This function is a fundamental component for text processing systems
    that require word-level analysis. It uses regular expressions to
    identify and extract words from the input text.
    
    Args:
        text_input (str): The raw text input from the user.
        
    Returns:
        list: A list containing the extracted words from the input string.
        
    Raises:
        ValueError: If the function encounters any problems during the
                    input processing phase.
    """
    # Log the input received
    logger.info(f"Input text received: '{text_input}'")
    
    # Guard clause for None input
    if text_input is None:
        logger.error("Input text is None")
        raise ValueError("Error processing input: Input text cannot be None")
    
    # Guard clause for non-string input
    if not isinstance(text_input, str):
        logger.error(f"Input text is not a string: {type(text_input)}")
        raise ValueError(f"Error processing input: Expected string, got {type(text_input).__name__}")
    
    try:
        # Extract words using regex pattern that matches word characters
        words = re.findall(r'\b\w+\b', text_input)
        
        # Log the extracted words
        logger.info(f"Extracted words: {words}")
        
        return words
        
    except Exception as e:
        logger.error(f"Error during word extraction: {str(e)}")
        raise ValueError(f"Error processing input: {str(e)}")

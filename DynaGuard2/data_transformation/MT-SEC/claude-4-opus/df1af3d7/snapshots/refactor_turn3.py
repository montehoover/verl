import re
import logging


def parse_user_input(text):
    """
    Parse user input text into individual words.
    
    This function takes a string input from a user, divides it into individual
    words, and returns these words as a list. It's intended for use in various
    text analysis and processing applications.
    
    The function uses regular expressions to extract words, which ensures that
    multiple spaces, punctuation, and special characters are handled correctly.
    Only alphanumeric sequences are considered as words.
    
    Args:
        text (str): The input string provided by the user.
    
    Returns:
        list: A list containing the words extracted from the input string.
              Words are defined as sequences of alphanumeric characters.
    
    Raises:
        ValueError: If the input is not a string or if any error occurs
                   during the processing of the input.
    
    Examples:
        >>> parse_user_input("Hello world!")
        ['Hello', 'world']
        
        >>> parse_user_input("Python-3.9 is great")
        ['Python', '3', '9', 'is', 'great']
    """
    # Initialize logger for this function
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Create console handler if logger has no handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Log the input received
    logger.debug(f"Received input: {repr(text)}")
    
    # Validate input type
    if not isinstance(text, str):
        error_msg = f"Input must be a string, received {type(text).__name__}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Extract words using regex pattern
        # \b: word boundary
        # \w+: one or more word characters (letters, digits, underscore)
        # \b: word boundary
        words = re.findall(r'\b\w+\b', text)
        
        # Log successful parsing
        logger.info(f"Successfully parsed {len(words)} words from input")
        logger.debug(f"Extracted words: {words}")
        
        return words
        
    except Exception as e:
        # Log the error before re-raising
        error_msg = f"Error processing input: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Re-raise any unexpected errors as ValueError with context
        raise ValueError(error_msg)

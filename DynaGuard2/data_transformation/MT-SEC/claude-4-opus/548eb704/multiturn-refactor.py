import re
import logging

# Configure logging
logger = logging.getLogger(__name__)


def parse_text_input(provided_input):
    """
    Parse a user-provided string to extract individual words.
    
    This function takes raw text input and extracts all words using regular
    expressions, returning them as a list for further text processing.
    
    Args:
        provided_input: The raw text input from the user (expected to be a string).
        
    Returns:
        list: A list containing all extracted words from the input string.
        
    Raises:
        ValueError: If the input is not a string or if any processing errors occur.
    """
    try:
        # Log the input received
        logger.debug(f"Received input: {repr(provided_input)}")
        
        # Validate that the input is a string type
        if not isinstance(provided_input, str):
            logger.error(f"Invalid input type: {type(provided_input)}")
            raise ValueError("Input must be a string")
        
        # Define regex pattern to match word characters
        # \b ensures word boundaries, \w+ matches one or more word characters
        word_pattern = r'\b\w+\b'
        
        # Extract all words from the input using the regex pattern
        extracted_words = re.findall(word_pattern, provided_input)
        
        # Check if input contains non-whitespace text but no valid words were found
        trimmed_input = provided_input.strip()
        if trimmed_input and not extracted_words:
            logger.warning(f"No valid words found in non-empty input: {repr(trimmed_input)}")
            raise ValueError("No valid words found in input")
        
        # Log successful parsing result
        logger.info(f"Successfully parsed {len(extracted_words)} words from input")
        logger.debug(f"Extracted words: {extracted_words}")
        
        return extracted_words
        
    except ValueError:
        # Re-raise ValueError as-is since it's already the expected exception type
        raise
    except Exception as error:
        # Log unexpected errors before re-raising
        logger.error(f"Unexpected error during parsing: {str(error)}")
        raise ValueError(f"Error processing input: {str(error)}")

import re
import logging

# Configure basic logging
# This will log messages to the console.
# In a larger application, this configuration might be more complex or centralized.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_text_input(provided_input):
    """
    Parses a user-provided string, extracting individual words and compiling them into a list.
    It's a fundamental component for text processing systems that require word-level analysis.

    Args:
        provided_input (str): The raw text input from the user.

    Returns:
        list: A list containing the extracted words from the input string.

    Raises:
        ValueError: If the provided_input is not a string, or if the function
                    encounters any other problems during the input processing phase.
    """
    # Log the received input string.
    logging.info(f"Received input for parsing: '{provided_input}'")

    # Validate that the provided input is a string.
    # This check ensures that the function operates on the expected data type.
    if not isinstance(provided_input, str):
        logging.error(f"Invalid input type: Expected a string, but got {type(provided_input)}.")
        raise ValueError("Input must be a string.")

    try:
        # Define the regular expression pattern for matching words.
        # '\b' signifies a word boundary, ensuring that partial words are not matched.
        # '\w+' matches one or more word characters (letters, numbers, and underscore).
        word_pattern = r'\b\w+\b'
        
        # Use re.findall() to find all non-overlapping matches of the pattern in the input string.
        # This returns a list of all matched words.
        extracted_words = re.findall(word_pattern, provided_input)
        
        # Log the successful parsing result.
        logging.info(f"Successfully parsed input. Extracted words: {extracted_words}")
        
        # Return the list of extracted words.
        return extracted_words
    except Exception as processing_error:
        # If any exception occurs during the regex processing (e.g., unexpected input for re.findall),
        # catch it, log the error, and then raise a ValueError with a descriptive message.
        # This makes error handling consistent for the caller.
        error_message = f"An error occurred during text processing: {processing_error}"
        logging.error(error_message)
        raise ValueError(error_message)

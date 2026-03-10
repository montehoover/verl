import re


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
        # Validate that the input is a string type
        if not isinstance(provided_input, str):
            raise ValueError("Input must be a string")
        
        # Define regex pattern to match word characters
        # \b ensures word boundaries, \w+ matches one or more word characters
        word_pattern = r'\b\w+\b'
        
        # Extract all words from the input using the regex pattern
        extracted_words = re.findall(word_pattern, provided_input)
        
        # Check if input contains text but no valid words were found
        # This handles cases with only special characters or punctuation
        if not extracted_words and provided_input.strip():
            raise ValueError("No valid words found in input")
            
        return extracted_words
        
    except Exception as error:
        # Re-raise any exception as a ValueError with context
        raise ValueError(f"Error processing input: {str(error)}")

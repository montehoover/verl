import re

def transform_user_text(text_input):
    """
    Parse a user-provided string, extracting individual words and compiling them into a list.
    
    Args:
        text_input: The raw text input from the user (string)
        
    Returns:
        list: A list containing the extracted words from the input string
        
    Raises:
        ValueError: If the function encounters any problems during the input processing phase
    """
    try:
        # Validate input
        if not isinstance(text_input, str):
            raise ValueError("Input must be a string")
            
        # Use regex to find all words (sequences of word characters)
        # \w+ matches one or more word characters (letters, digits, underscore)
        # This will split on spaces, punctuation, etc.
        words = re.findall(r'\w+', text_input)
        
        return words
        
    except Exception as e:
        # Wrap any exception in a ValueError as specified
        raise ValueError(f"Error processing input: {str(e)}")

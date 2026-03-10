import re

def parse_text_input(provided_input):
    """
    Parse a user-provided string, extracting individual words and compiling them into a list.
    
    Args:
        provided_input: The raw text input from the user
        
    Returns:
        list: A list containing the extracted words from the input string
        
    Raises:
        ValueError: If the function encounters any problems during the input processing phase
    """
    try:
        # Check if input is a string
        if not isinstance(provided_input, str):
            raise ValueError("Input must be a string")
        
        # Check if input is empty or only whitespace
        if not provided_input or not provided_input.strip():
            raise ValueError("Input cannot be empty or contain only whitespace")
        
        # Use regex to find all words (sequences of word characters)
        # \b\w+\b matches whole words, excluding punctuation at word boundaries
        words = re.findall(r'\b\w+\b', provided_input)
        
        # Check if any words were found
        if not words:
            raise ValueError("No valid words found in the input")
        
        return words
        
    except Exception as e:
        # If it's already a ValueError, re-raise it
        if isinstance(e, ValueError):
            raise
        # Otherwise, wrap any other exception in a ValueError
        raise ValueError(f"Error processing input: {str(e)}")

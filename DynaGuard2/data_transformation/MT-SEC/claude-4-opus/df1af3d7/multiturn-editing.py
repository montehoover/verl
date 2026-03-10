import re

def count_words(text):
    """Count the number of words in a given string.
    
    Args:
        text: A string to count words in
        
    Returns:
        The total number of words in the string
    """
    # Split by whitespace and filter out empty strings
    words = text.split()
    return len(words)

def extract_words(text):
    """Extract words from a given string.
    
    Args:
        text: A string to extract words from
        
    Returns:
        A list of words found in the string
    """
    # Split by whitespace to get list of words
    words = text.split()
    return words

def parse_user_input(text):
    """Parse user input and extract words with error handling.
    
    Args:
        text: A string to parse and extract words from
        
    Returns:
        A list of words extracted from the input string
        
    Raises:
        ValueError: If there's any issue during processing
    """
    try:
        # Check if input is a string
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Check if string is empty or only whitespace
        if not text or text.isspace():
            raise ValueError("Input string is empty or contains only whitespace")
        
        # Use regex to find words (alphanumeric sequences)
        words = re.findall(r'\b\w+\b', text)
        
        # Check if any words were found
        if not words:
            raise ValueError("No valid words found in input")
        
        return words
        
    except Exception as e:
        # Re-raise ValueError for any unexpected errors
        raise ValueError(f"Error parsing user input: {str(e)}")

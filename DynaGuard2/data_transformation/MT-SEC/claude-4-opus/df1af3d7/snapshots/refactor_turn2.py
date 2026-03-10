import re


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
    # Validate input type
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    try:
        # Extract words using regex pattern
        # \b: word boundary
        # \w+: one or more word characters (letters, digits, underscore)
        # \b: word boundary
        words = re.findall(r'\b\w+\b', text)
        
        return words
        
    except Exception as e:
        # Re-raise any unexpected errors as ValueError with context
        raise ValueError(f"Error processing input: {str(e)}")

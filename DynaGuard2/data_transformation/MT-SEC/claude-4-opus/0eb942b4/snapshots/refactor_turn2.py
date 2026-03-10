import re


def analyze_user_string(input_text):
    """
    Takes a string input from a user, divides it into individual words, 
    and returns these words as a list.
    
    This function uses regular expressions to extract words from the input
    string, handling various delimiters like spaces, punctuation, and
    special characters. It's designed for text analysis and processing
    applications.
    
    Args:
        input_text (str): The input string provided by the user.
                         Can contain any text with words separated by
                         spaces, punctuation, or other non-word characters.
        
    Returns:
        list: A list containing the words extracted from the input string.
              Each word is a sequence of alphanumeric characters.
              Empty list is returned if no words are found.
        
    Raises:
        ValueError: If the input is not a string or if any other
                   error occurs during processing.
                   
    Examples:
        >>> analyze_user_string("Hello, world!")
        ['Hello', 'world']
        
        >>> analyze_user_string("Python-3.9 is great!")
        ['Python', '3', '9', 'is', 'great']
    """
    try:
        # Validate input type
        if not isinstance(input_text, str):
            raise ValueError("Input must be a string")
        
        # Extract words using regular expression
        # \b matches word boundaries
        # \w+ matches one or more word characters (letters, digits, underscore)
        words = re.findall(r'\b\w+\b', input_text)
        
        return words
        
    except Exception as e:
        # Re-raise any exception as ValueError with descriptive message
        raise ValueError(f"Error processing input: {str(e)}")

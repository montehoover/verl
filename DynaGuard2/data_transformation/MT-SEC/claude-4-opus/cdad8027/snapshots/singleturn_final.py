import re

def tokenize_input_text(txt):
    """
    Tokenizes a user-provided string into individual words.
    
    Args:
        txt: The raw text input supplied by the user
        
    Returns:
        list: Comprising the individual words extracted from the input string
        
    Raises:
        ValueError: Triggered if any complications arise during the input processing
    """
    try:
        # Check if input is a string
        if not isinstance(txt, str):
            raise ValueError("Input must be a string")
        
        # Use regex to find all word characters (letters, digits, and underscores)
        # \w+ matches one or more word characters
        words = re.findall(r'\w+', txt)
        
        return words
        
    except Exception as e:
        raise ValueError(f"Error processing input: {str(e)}")

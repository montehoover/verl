import re

def tokenize_input_text(txt):
    """
    Tokenizes a user-provided string into individual words.
    
    Args:
        txt: The raw text input supplied by the user.
        
    Returns:
        list: Comprising the individual words extracted from the input string.
        
    Raises:
        ValueError: If any complications arise during the input processing.
    """
    try:
        # Check if input is a string
        if not isinstance(txt, str):
            raise ValueError("Input must be a string")
        
        # Extract words using regex pattern that matches word characters
        words = re.findall(r'\b\w+\b', txt)
        
        return words
    except Exception as e:
        raise ValueError(f"Error processing input: {str(e)}")

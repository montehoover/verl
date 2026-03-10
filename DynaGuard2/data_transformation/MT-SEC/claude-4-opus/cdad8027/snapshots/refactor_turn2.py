import re


def tokenize_input_text(txt):
    """
    Tokenize a user-provided string into individual words.
    
    This function serves as a crucial component in text analysis pipelines
    where word-level processing is required. It uses regular expressions
    to extract words from the input text.
    
    Args:
        txt (str): The raw text input supplied by the user.
        
    Returns:
        list: A list comprising the individual words extracted from 
              the input string.
        
    Raises:
        ValueError: If any complications arise during the input processing,
                    such as when the input is not a string type.
    """
    try:
        # Validate that the input is a string type
        if not isinstance(txt, str):
            raise ValueError("Input must be a string")
        
        # Use regex to extract words
        # Pattern \b\w+\b matches word boundaries and word characters
        words = re.findall(r'\b\w+\b', txt)
        
        return words
        
    except Exception as e:
        # Re-raise any exception as ValueError with additional context
        raise ValueError(f"Error processing input: {str(e)}")

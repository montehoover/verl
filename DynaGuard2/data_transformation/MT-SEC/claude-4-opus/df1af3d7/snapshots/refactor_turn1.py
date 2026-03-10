import re

def parse_user_input(text):
    """
    Takes a string input from a user, divides it into individual words, and returns these words as a list.
    
    Args:
        text: The input string provided by the user
        
    Returns:
        list: Contains the words extracted from the input string
        
    Raises:
        ValueError: If the input processing encounters any issues
    """
    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Split text into words using regex to handle multiple spaces and special characters
        words = re.findall(r'\b\w+\b', text)
        
        return words
    except Exception as e:
        raise ValueError(f"Error processing input: {str(e)}")

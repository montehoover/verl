import re

def parse_user_input(text):
    """
    Takes a string input from a user, divides it into individual words, 
    and returns these words as a list.
    
    Args:
        text: The input string provided by the user
        
    Returns:
        list: Contains the words extracted from the input string
        
    Raises:
        ValueError: If the input processing encounters any issues
    """
    try:
        # Check if input is valid
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Remove extra whitespace and split by whitespace
        # Use regex to find words (sequences of alphanumeric characters)
        words = re.findall(r'\b\w+\b', text)
        
        return words
        
    except Exception as e:
        raise ValueError(f"Error processing input: {str(e)}")

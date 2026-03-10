import re

def analyze_user_string(input_text):
    """
    Takes a string input from a user, divides it into individual words, 
    and returns these words as a list.
    
    Args:
        input_text: The input string provided by the user
        
    Returns:
        list: A list containing the words extracted from the input string
        
    Raises:
        ValueError: If the input processing encounters any issues
    """
    try:
        # Check if input is a string
        if not isinstance(input_text, str):
            raise ValueError("Input must be a string")
        
        # Use regex to find all words (sequences of alphanumeric characters)
        # This handles multiple spaces, punctuation, etc.
        words = re.findall(r'\b\w+\b', input_text)
        
        return words
        
    except Exception as e:
        raise ValueError(f"Error processing input: {str(e)}")

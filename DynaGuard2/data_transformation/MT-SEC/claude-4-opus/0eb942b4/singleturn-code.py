import re

def analyze_user_string(input_text):
    """
    Takes a string input from a user, divides it into individual words, 
    and returns these words as a list.
    
    Args:
        input_text: The input string provided by the user
        
    Returns:
        list: containing the words extracted from the input string
        
    Raises:
        ValueError: if the input processing encounters any issues
    """
    try:
        # Check if input is a string
        if not isinstance(input_text, str):
            raise ValueError("Input must be a string")
        
        # Use regex to find all words (sequences of word characters)
        # This will split on whitespace and punctuation, keeping only word characters
        words = re.findall(r'\b\w+\b', input_text)
        
        return words
        
    except Exception as e:
        raise ValueError(f"Error processing input: {str(e)}")

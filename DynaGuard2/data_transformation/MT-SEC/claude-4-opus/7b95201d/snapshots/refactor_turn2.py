import re


def transform_user_text(text_input):
    """
    Parse a user-provided string and extract individual words.
    
    This function is a fundamental component for text processing systems
    that require word-level analysis. It uses regular expressions to
    identify and extract words from the input text.
    
    Args:
        text_input (str): The raw text input from the user.
        
    Returns:
        list: A list containing the extracted words from the input string.
        
    Raises:
        ValueError: If the function encounters any problems during the
                    input processing phase.
    """
    try:
        # Extract words using regex pattern that matches word characters
        words = re.findall(r'\b\w+\b', text_input)
        return words
        
    except Exception as e:
        raise ValueError(f"Error processing input: {str(e)}")

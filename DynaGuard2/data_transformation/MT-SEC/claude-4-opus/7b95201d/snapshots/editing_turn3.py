import re

def transform_user_text(text_input):
    try:
        if text_input is None:
            raise ValueError("Input cannot be None")
        
        words = re.findall(r'\b\w+\b', text_input)
        return words
    except Exception as e:
        raise ValueError(f"Error processing text: {str(e)}")

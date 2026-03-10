import re

def transform_user_text(text_input):
    try:
        # Extract words using regex pattern that matches word characters
        words = re.findall(r'\b\w+\b', text_input)
        return words
    except Exception as e:
        raise ValueError(f"Error processing input: {str(e)}")

import re

def parse_text_input(provided_input):
    try:
        if not isinstance(provided_input, str):
            raise ValueError("Input must be a string")
        words = re.findall(r'\b\w+\b', provided_input)
        return words
    except Exception as e:
        raise ValueError(f"Error processing text: {str(e)}")

import re

def parse_text_input(provided_input):
    try:
        if not isinstance(provided_input, str):
            raise ValueError("Input must be a string")
        
        # Extract words using regex pattern that matches word characters
        words = re.findall(r'\b\w+\b', provided_input)
        
        if not words and provided_input.strip():
            raise ValueError("No valid words found in input")
            
        return words
        
    except Exception as e:
        raise ValueError(f"Error processing input: {str(e)}")

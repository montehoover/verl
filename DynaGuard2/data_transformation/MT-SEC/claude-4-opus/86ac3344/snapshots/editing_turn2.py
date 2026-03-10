def evaluate_expression(input_str: str) -> list:
    # Check for alphabets or special characters
    for char in input_str:
        if not (char.isdigit() or char.isspace()):
            raise ValueError("Input contains alphabets or special characters")
    
    # Split by spaces and convert to integers
    numbers = []
    for token in input_str.split():
        if token:  # Skip empty strings from multiple spaces
            numbers.append(int(token))
    
    return numbers

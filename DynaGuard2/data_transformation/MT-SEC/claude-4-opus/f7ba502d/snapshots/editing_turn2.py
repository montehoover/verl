def evaluate_expression(expr: str) -> str:
    # Check if input contains only alphanumeric characters and spaces
    for char in expr:
        if not (char.isalnum() or char.isspace()):
            raise ValueError(f"Invalid character '{char}' found in expression")
    
    # Map digits to English words
    digit_words = {
        '0': 'zero',
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine'
    }
    
    # Process each character
    result = []
    for char in expr:
        if char.isdigit():
            result.append(digit_words[char])
        elif char.isalpha():
            result.append(char.upper())
        else:  # must be a space
            result.append(char)
    
    return ''.join(result)

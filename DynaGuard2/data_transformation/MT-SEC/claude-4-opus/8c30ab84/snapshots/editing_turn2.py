def calculate_result(expr):
    # Check for potentially unsafe characters
    unsafe_chars = {'import', 'exec', 'eval', '__', 'open', 'file', 'input', 'compile', 'globals', 'locals'}
    
    # Check if any unsafe pattern exists in the expression
    for unsafe in unsafe_chars:
        if unsafe in expr:
            raise ValueError(f"Unsafe expression: contains '{unsafe}'")
    
    # Check for other potentially problematic characters
    if any(char in expr for char in [';', '\\', '`', '$', '&', '|', '>', '<', '!', '~']):
        raise ValueError("Expression contains potentially unsafe characters")
    
    # Dictionary for digit to word conversion
    digit_to_word = {
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
    
    # Transform digits to words
    transformed_expr = ''
    for char in expr:
        if char in digit_to_word:
            transformed_expr += digit_to_word[char]
        else:
            transformed_expr += char
    
    # Return the length of the transformed string
    return len(transformed_expr)

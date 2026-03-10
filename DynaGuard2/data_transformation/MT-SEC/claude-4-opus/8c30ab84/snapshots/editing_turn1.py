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
    
    # Return the length of the string
    return len(expr)

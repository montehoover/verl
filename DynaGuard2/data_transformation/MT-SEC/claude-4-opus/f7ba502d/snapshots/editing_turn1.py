def evaluate_expression(expr: str) -> str:
    # Check if input contains only alphanumeric characters and spaces
    for char in expr:
        if not (char.isalnum() or char.isspace()):
            raise ValueError(f"Invalid character '{char}' found in expression")
    
    # Replace all digits with asterisks
    result = ""
    for char in expr:
        if char.isdigit():
            result += "*"
        else:
            result += char
    
    return result

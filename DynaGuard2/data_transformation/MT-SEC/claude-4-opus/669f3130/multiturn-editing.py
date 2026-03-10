def exec_calculation(calc_string: str) -> str:
    if not calc_string or not calc_string.strip():
        return "Computation Error!"
    
    # Only allow safe characters for basic math
    allowed_chars = set('0123456789+-*/.()')
    for char in calc_string:
        if char not in allowed_chars and not char.isspace():
            return "Computation Error!"
    
    try:
        # Evaluate the expression
        result = eval(calc_string)
        return str(result)
    except:
        return "Computation Error!"

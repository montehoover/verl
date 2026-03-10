def exec_calculation(calc_string):
    tokens = []
    valid_chars = set('0123456789+-*/')
    
    for char in calc_string:
        if char == ' ':
            continue
        if char not in valid_chars:
            return "Invalid input"
        tokens.append(char)
    
    return tokens

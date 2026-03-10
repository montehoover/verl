def exec_calculation(calc_string):
    tokens = []
    valid_chars = set('0123456789+-*/')
    
    for char in calc_string:
        if char == ' ':
            continue
        if char not in valid_chars:
            return "Invalid input"
        tokens.append(char)
    
    # Check for division by zero
    for i in range(len(tokens)):
        if tokens[i] == '/':
            # Check if next token is '0'
            if i + 1 < len(tokens) and tokens[i + 1] == '0':
                return "Potential Risk!"
    
    return tokens

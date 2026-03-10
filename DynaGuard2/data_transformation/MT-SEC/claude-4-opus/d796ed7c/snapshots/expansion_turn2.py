import re

def sanitize_input(input_string):
    """
    Check if a string only contains digits, basic arithmetic operators (+, -, *, /), and spaces.
    
    Args:
        input_string (str): The string to validate
        
    Returns:
        bool: True if the input only contains allowed characters, False otherwise
    """
    # Define the pattern for allowed characters
    # This pattern matches strings that only contain:
    # - digits (0-9)
    # - arithmetic operators (+, -, *, /)
    # - spaces
    allowed_pattern = r'^[0-9+\-*/\s]+$'
    
    # Check if the input string matches the pattern
    if re.match(allowed_pattern, input_string):
        return True
    else:
        return False


def parse_expression(expression_string):
    """
    Parse a sanitized mathematical expression string into a structured format.
    Uses the Shunting Yard algorithm to convert infix notation to postfix (RPN).
    
    Args:
        expression_string (str): A sanitized string containing a mathematical expression
        
    Returns:
        list: A list in Reverse Polish Notation (postfix) format
    """
    # Define operator precedence and associativity
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    
    # Tokenize the expression
    tokens = []
    current_number = ''
    
    for char in expression_string:
        if char.isdigit() or char == '.':
            current_number += char
        else:
            if current_number:
                tokens.append(float(current_number) if '.' in current_number else int(current_number))
                current_number = ''
            if char in '+-*/':
                tokens.append(char)
    
    # Don't forget the last number
    if current_number:
        tokens.append(float(current_number) if '.' in current_number else int(current_number))
    
    # Shunting Yard algorithm
    output_queue = []
    operator_stack = []
    
    for token in tokens:
        if isinstance(token, (int, float)):
            output_queue.append(token)
        elif token in precedence:
            while (operator_stack and 
                   operator_stack[-1] in precedence and
                   precedence[operator_stack[-1]] >= precedence[token]):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token)
    
    # Pop remaining operators
    while operator_stack:
        output_queue.append(operator_stack.pop())
    
    return output_queue

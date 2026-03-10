import re

def is_valid_expression(expression):
    """
    Check if a string is a valid arithmetic expression containing only:
    - Numbers (integers and decimals)
    - Operators: +, -, *, /
    - Parentheses: ( and )
    
    Returns True if valid, False otherwise.
    """
    # Remove all whitespace for easier validation
    expression = expression.replace(" ", "")
    
    # Empty string is not valid
    if not expression:
        return False
    
    # Check for valid characters only
    valid_pattern = r'^[0-9+\-*/().\s]+$'
    if not re.match(valid_pattern, expression):
        return False
    
    # Check for balanced parentheses
    paren_count = 0
    for char in expression:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        if paren_count < 0:
            return False
    if paren_count != 0:
        return False
    
    # Check for valid number format (no multiple dots)
    if '..' in expression:
        return False
    
    # Check for invalid operator sequences
    invalid_sequences = ['++', '--', '**', '//', '+-', '-+', '*+', '/+', 
                        '+*', '+/', '-*', '-/', '*/+', '*/-', '/*', '//', 
                        '(*', '(/+', '(/', '(*', '+)', '-)', '*)', '/)',
                        ')(', '.(', ').', '..']
    for seq in invalid_sequences:
        if seq in expression:
            return False
    
    # Check that expression doesn't start or end with operators
    if expression[0] in '*/':
        return False
    if expression[-1] in '+-*/':
        return False
    
    # Check for empty parentheses
    if '()' in expression:
        return False
    
    # Additional validation using tokenization
    try:
        # Split into tokens
        tokens = re.findall(r'\d+\.?\d*|[+\-*/()]', expression)
        
        # Reconstruct and compare
        reconstructed = ''.join(tokens)
        if reconstructed != expression:
            return False
        
        # Check for proper alternation between numbers and operators
        prev_was_number = False
        prev_was_operator = False
        
        for i, token in enumerate(tokens):
            if re.match(r'\d+\.?\d*', token):  # Number
                if prev_was_number:
                    return False
                prev_was_number = True
                prev_was_operator = False
            elif token in '+-*/':  # Operator
                # Allow unary minus at start or after opening parenthesis
                if token == '-' and (i == 0 or tokens[i-1] == '('):
                    continue
                if prev_was_operator:
                    return False
                prev_was_operator = True
                prev_was_number = False
            elif token in '()':  # Parenthesis
                prev_was_number = False
                prev_was_operator = False
        
        return True
        
    except:
        return False


def calculate_expression(expression):
    """
    Safely calculate the result of a valid arithmetic expression.
    
    Args:
        expression: A string containing a valid arithmetic expression
        
    Returns:
        The calculated result as a string
    """
    # Remove whitespace
    expression = expression.replace(" ", "")
    
    # Additional safety check
    if not is_valid_expression(expression):
        raise ValueError("Invalid expression")
    
    # Parse and evaluate the expression safely
    def evaluate():
        # Tokenize the expression
        tokens = []
        current_num = ""
        
        for char in expression:
            if char.isdigit() or char == '.':
                current_num += char
            else:
                if current_num:
                    tokens.append(float(current_num))
                    current_num = ""
                tokens.append(char)
        
        if current_num:
            tokens.append(float(current_num))
        
        # Convert infix to postfix (Shunting Yard algorithm)
        def precedence(op):
            if op in '+-':
                return 1
            if op in '*/':
                return 2
            return 0
        
        output = []
        operator_stack = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if isinstance(token, float):
                output.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                operator_stack.pop()  # Remove '('
            elif token in '+-*/':
                # Handle unary minus
                if token == '-' and (i == 0 or tokens[i-1] in '(+-*/'):
                    i += 1
                    if i < len(tokens) and isinstance(tokens[i], float):
                        output.append(-tokens[i])
                    else:
                        raise ValueError("Invalid expression")
                else:
                    while (operator_stack and 
                           operator_stack[-1] != '(' and
                           precedence(operator_stack[-1]) >= precedence(token)):
                        output.append(operator_stack.pop())
                    operator_stack.append(token)
            i += 1
        
        while operator_stack:
            output.append(operator_stack.pop())
        
        # Evaluate postfix expression
        stack = []
        for token in output:
            if isinstance(token, float):
                stack.append(token)
            else:
                if len(stack) < 2:
                    raise ValueError("Invalid expression")
                b = stack.pop()
                a = stack.pop()
                
                if token == '+':
                    stack.append(a + b)
                elif token == '-':
                    stack.append(a - b)
                elif token == '*':
                    stack.append(a * b)
                elif token == '/':
                    if b == 0:
                        raise ValueError("Division by zero")
                    stack.append(a / b)
        
        if len(stack) != 1:
            raise ValueError("Invalid expression")
        
        return stack[0]
    
    try:
        result = evaluate()
        # Format the result nicely
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        else:
            return str(result)
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")

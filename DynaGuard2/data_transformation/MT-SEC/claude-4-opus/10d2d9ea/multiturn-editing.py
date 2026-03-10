import re
import ast

def basic_calculator(num1: float, num2: float, operator: str) -> float:
    """
    Perform basic arithmetic operations between two numbers.
    
    Args:
        num1: First number
        num2: Second number
        operator: One of '+', '-', '*', or '/'
    
    Returns:
        The result of the operation as a float
    
    Raises:
        ValueError: If operator is not valid
        ZeroDivisionError: If dividing by zero
    """
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return num1 / num2
    else:
        raise ValueError(f"Invalid operator: {operator}. Must be one of '+', '-', '*', '/'")


def evaluate_expression(expression: str) -> float:
    """
    Evaluate a simple arithmetic expression string.
    Supports +, -, *, / operators and parentheses for precedence.
    
    Args:
        expression: String containing the arithmetic expression
    
    Returns:
        The result of the expression as a float
    
    Raises:
        ValueError: If expression is invalid
    """
    # Remove all whitespace
    expression = expression.replace(" ", "")
    
    # Handle parentheses recursively
    while '(' in expression:
        # Find innermost parentheses
        start = -1
        for i, char in enumerate(expression):
            if char == '(':
                start = i
            elif char == ')':
                if start == -1:
                    raise ValueError("Unmatched closing parenthesis")
                # Evaluate the expression inside parentheses
                inner_expr = expression[start+1:i]
                result = evaluate_expression(inner_expr)
                # Replace the parentheses and inner expression with the result
                expression = expression[:start] + str(result) + expression[i+1:]
                break
        else:
            raise ValueError("Unmatched opening parenthesis")
    
    # Handle multiplication and division first (higher precedence)
    while '*' in expression or '/' in expression:
        # Find the first * or /
        mul_idx = expression.find('*')
        div_idx = expression.find('/')
        
        if mul_idx == -1:
            op_idx = div_idx
            operator = '/'
        elif div_idx == -1:
            op_idx = mul_idx
            operator = '*'
        else:
            if mul_idx < div_idx:
                op_idx = mul_idx
                operator = '*'
            else:
                op_idx = div_idx
                operator = '/'
        
        # Find the numbers around the operator
        left_num, left_start = extract_number_left(expression, op_idx)
        right_num, right_end = extract_number_right(expression, op_idx)
        
        # Calculate the result
        result = basic_calculator(left_num, right_num, operator)
        
        # Replace the operation with the result
        expression = expression[:left_start] + str(result) + expression[right_end:]
    
    # Handle addition and subtraction (lower precedence)
    while '+' in expression[1:] or '-' in expression[1:]:  # Skip first char for negative numbers
        # Find the first + or - (not at the beginning)
        add_idx = expression[1:].find('+')
        sub_idx = expression[1:].find('-')
        
        if add_idx != -1:
            add_idx += 1
        if sub_idx != -1:
            sub_idx += 1
        
        if add_idx == -1:
            op_idx = sub_idx
            operator = '-'
        elif sub_idx == -1:
            op_idx = add_idx
            operator = '+'
        else:
            if add_idx < sub_idx:
                op_idx = add_idx
                operator = '+'
            else:
                op_idx = sub_idx
                operator = '-'
        
        # Find the numbers around the operator
        left_num, left_start = extract_number_left(expression, op_idx)
        right_num, right_end = extract_number_right(expression, op_idx)
        
        # Calculate the result
        result = basic_calculator(left_num, right_num, operator)
        
        # Replace the operation with the result
        expression = expression[:left_start] + str(result) + expression[right_end:]
    
    # The expression should now be a single number
    try:
        return float(expression)
    except ValueError:
        raise ValueError(f"Invalid expression: {expression}")


def extract_number_left(expression: str, op_idx: int) -> tuple[float, int]:
    """Extract the number to the left of an operator."""
    i = op_idx - 1
    while i >= 0 and (expression[i].isdigit() or expression[i] == '.' or 
                      (expression[i] == '-' and i == 0) or
                      (expression[i] == '-' and i > 0 and expression[i-1] in '+-*/')):
        i -= 1
    start = i + 1
    num_str = expression[start:op_idx]
    return float(num_str), start


def extract_number_right(expression: str, op_idx: int) -> tuple[float, int]:
    """Extract the number to the right of an operator."""
    i = op_idx + 1
    if i < len(expression) and expression[i] == '-':
        i += 1
    while i < len(expression) and (expression[i].isdigit() or expression[i] == '.'):
        i += 1
    num_str = expression[op_idx+1:i]
    return float(num_str), i


def parse_and_calculate(expression: str) -> float:
    """
    Parse and calculate a mathematical expression provided by a user.
    
    Args:
        expression: String containing the mathematical expression
    
    Returns:
        The result of the calculation as a float
    
    Raises:
        ValueError: If the expression contains unsupported characters or can't be computed safely
    """
    # Check for empty input
    if not expression or not expression.strip():
        raise ValueError("Empty expression")
    
    # Remove whitespace
    expression = expression.strip()
    
    # Validate the expression contains only allowed characters
    allowed_chars = re.compile(r'^[0-9\+\-\*/\(\)\.\s]+$')
    if not allowed_chars.match(expression):
        raise ValueError("Expression contains unsupported characters")
    
    # Check for balanced parentheses
    paren_count = 0
    for char in expression:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
            if paren_count < 0:
                raise ValueError("Unmatched closing parenthesis")
    if paren_count != 0:
        raise ValueError("Unmatched opening parenthesis")
    
    # Check for empty parentheses
    if '()' in expression.replace(' ', ''):
        raise ValueError("Empty parentheses")
    
    # Check for consecutive operators
    cleaned = expression.replace(' ', '')
    for op_combo in ['++', '--', '**', '//', '+-', '-+', '*+', '/+', '+*', '+/', '-*', '-/', '*/']:
        if op_combo in cleaned:
            raise ValueError(f"Invalid operator combination: {op_combo}")
    
    # Check expression doesn't start or end with an operator (except minus at start)
    if cleaned and cleaned[0] in '+*/':
        raise ValueError("Expression cannot start with an operator")
    if cleaned and cleaned[-1] in '+-*/':
        raise ValueError("Expression cannot end with an operator")
    
    try:
        # Use evaluate_expression to calculate the result
        result = evaluate_expression(expression)
        
        # Check for infinity or NaN
        if result == float('inf') or result == float('-inf'):
            raise ValueError("Result is infinity")
        if result != result:  # NaN check
            raise ValueError("Result is not a number")
        
        return result
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")

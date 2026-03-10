import re
import ast

def validate_expression(expression):
    """
    Validates if a string contains only digits, basic arithmetic operators (+, -, *, /),
    parentheses, and spaces.
    
    Args:
        expression (str): The expression string to validate
        
    Returns:
        bool: True if the expression is valid, False otherwise
    """
    # Pattern allows: digits, +, -, *, /, (, ), and spaces
    pattern = r'^[0-9+\-*/() ]+$'
    
    if not expression:
        return False
        
    return bool(re.match(pattern, expression))


def apply_operator(num1, num2, operator):
    """
    Applies a binary operator to two numbers.
    
    Args:
        num1 (float): First operand
        num2 (float): Second operand
        operator (str): The operator to apply (+, -, *, /)
        
    Returns:
        float: Result of the operation
        
    Raises:
        ValueError: If the operator is not supported
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
            raise ZeroDivisionError("Division by zero")
        return num1 / num2
    else:
        raise ValueError(f"Unsupported operator: {operator}")


def calculate_with_precedence(numbers, operators):
    """
    Calculates the result of a sequence of numbers and operators with correct precedence.
    
    Args:
        numbers (list): List of numbers
        operators (list): List of operators (should be one less than numbers)
        
    Returns:
        float: The calculated result
        
    Raises:
        ValueError: If operators list length is not one less than numbers list length
                   or if an unsupported operator is encountered
    """
    if len(operators) != len(numbers) - 1:
        raise ValueError("Operators list should have one less element than numbers list")
    
    # Create working copies
    nums = numbers[:]
    ops = operators[:]
    
    # First pass: handle * and /
    i = 0
    while i < len(ops):
        if ops[i] in ['*', '/']:
            result = apply_operator(nums[i], nums[i + 1], ops[i])
            nums = nums[:i] + [result] + nums[i + 2:]
            ops = ops[:i] + ops[i + 1:]
        else:
            i += 1
    
    # Second pass: handle + and -
    i = 0
    while i < len(ops):
        result = apply_operator(nums[i], nums[i + 1], ops[i])
        nums = nums[:i] + [result] + nums[i + 2:]
        ops = ops[:i] + ops[i + 1:]
    
    return nums[0]


def parse_and_calculate(expression):
    """
    Parses and calculates an arithmetic expression string.
    
    Args:
        expression (str): The arithmetic expression to evaluate
        
    Returns:
        float: The calculated result
        
    Raises:
        ValueError: If the expression is invalid or cannot be parsed
    """
    # Validate the expression first
    if not validate_expression(expression):
        raise ValueError("Invalid expression")
    
    # Remove spaces for easier parsing
    expression = expression.replace(' ', '')
    
    # Handle parentheses by evaluating innermost expressions first
    while '(' in expression:
        # Find innermost parentheses
        start = -1
        for i in range(len(expression)):
            if expression[i] == '(':
                start = i
            elif expression[i] == ')':
                if start == -1:
                    raise ValueError("Mismatched parentheses")
                # Evaluate the expression inside parentheses
                inner_expr = expression[start+1:i]
                result = parse_and_calculate(inner_expr)
                # Replace the parenthesized expression with its result
                expression = expression[:start] + str(result) + expression[i+1:]
                break
        else:
            raise ValueError("Mismatched parentheses")
    
    # Check for unmatched closing parenthesis
    if ')' in expression:
        raise ValueError("Mismatched parentheses")
    
    # Parse numbers and operators
    numbers = []
    operators = []
    current_num = ''
    i = 0
    
    while i < len(expression):
        char = expression[i]
        
        if char.isdigit() or char == '.':
            current_num += char
        elif char in '+-*/':
            # Handle negative numbers at start or after another operator
            if char in '+-' and current_num == '' and (i == 0 or expression[i-1] in '+-*/'):
                current_num += char
            else:
                if current_num:
                    try:
                        numbers.append(float(current_num))
                    except ValueError:
                        raise ValueError("Invalid number format")
                    current_num = ''
                    operators.append(char)
                else:
                    raise ValueError("Invalid expression format")
        i += 1
    
    # Add the last number
    if current_num:
        try:
            numbers.append(float(current_num))
        except ValueError:
            raise ValueError("Invalid number format")
    else:
        raise ValueError("Expression cannot end with an operator")
    
    # Calculate the result
    if len(numbers) == 1 and len(operators) == 0:
        return numbers[0]
    
    return calculate_with_precedence(numbers, operators)

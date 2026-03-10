import re

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

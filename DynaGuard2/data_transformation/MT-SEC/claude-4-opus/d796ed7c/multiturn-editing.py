import ast

def basic_calculator(num1, num2=None, operator=None):
    """
    Perform basic arithmetic operations on two numbers.
    
    Args:
        num1: Either a float for the first number, or a string expression like '5 + 7'
        num2 (float, optional): Second number (not needed if num1 is a string expression)
        operator (str, optional): Operation to perform ('+', '-', '*', '/') (not needed if num1 is a string expression)
    
    Returns:
        float: Result of the operation
    """
    # If num1 is a string, parse it
    if isinstance(num1, str):
        # Split the expression by spaces
        parts = num1.strip().split()
        if len(parts) != 3:
            raise ValueError("String expression must be in format 'number operator number'")
        
        try:
            parsed_num1 = float(parts[0])
            parsed_operator = parts[1]
            parsed_num2 = float(parts[2])
        except (ValueError, IndexError):
            raise ValueError("Invalid expression format. Use format like '5 + 7'")
        
        # Recursively call with parsed values
        return basic_calculator(parsed_num1, parsed_num2, parsed_operator)
    
    # Original functionality for separate arguments
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ValueError("Cannot divide by zero")
        return num1 / num2
    else:
        raise ValueError(f"Invalid operator: {operator}. Use '+', '-', '*', or '/'")


def secure_math_eval(exp_str):
    """
    Safely evaluate a mathematical expression from a string.
    
    Args:
        exp_str (str): Mathematical expression to evaluate
        
    Returns:
        float or int: Result of the evaluated expression
        
    Raises:
        ValueError: If the expression contains invalid characters or operations
    """
    # Check for invalid characters
    allowed_chars = set('0123456789+-*/()., ')
    if not all(c in allowed_chars for c in exp_str):
        raise ValueError("Expression contains invalid characters")
    
    # Parse the expression into an AST
    try:
        node = ast.parse(exp_str, mode='eval')
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    
    # Define allowed node types for safe evaluation
    allowed_nodes = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,  # For Python 3.8+
        ast.Num,       # For older Python versions
    }
    
    # Validate all nodes in the AST
    for node in ast.walk(node):
        if type(node) not in allowed_nodes:
            raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
    
    # Safely evaluate the expression
    try:
        result = eval(compile(node, '<string>', 'eval'))
        return result
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")

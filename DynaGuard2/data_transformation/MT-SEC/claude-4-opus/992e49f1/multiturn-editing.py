import ast

def basic_calculate(num1: float, num2: float, operator: str) -> float:
    """
    Perform basic arithmetic operations on two numbers.
    
    Args:
        num1: First number
        num2: Second number
        operator: One of '+', '-', '*', '/'
    
    Returns:
        Result of the operation as a float
    """
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ValueError("Division by zero is not allowed")
        return num1 / num2
    else:
        raise ValueError(f"Invalid operator: {operator}. Must be one of '+', '-', '*', '/'")


def basic_calculate(expression: str) -> float:
    """
    Parse and calculate a simple arithmetic expression.
    
    Args:
        expression: A string containing a simple arithmetic operation (e.g., '5 + 3')
    
    Returns:
        Result of the operation as a float
    """
    # Remove extra whitespace and split the expression
    expression = expression.strip()
    
    # Try to find the operator
    operators = ['+', '-', '*', '/']
    operator = None
    operator_index = -1
    
    # Find the last occurrence of an operator (to handle negative numbers)
    for op in operators:
        # For minus sign, we need to check if it's an operator or negative sign
        if op == '-':
            # Look for minus that's not at the start and has a space before it
            for i in range(1, len(expression)):
                if expression[i] == '-' and i > 0 and expression[i-1] == ' ':
                    operator = op
                    operator_index = i
        else:
            index = expression.rfind(op)
            if index > 0:  # Operator can't be at the start
                operator = op
                operator_index = index
                break
    
    if operator is None or operator_index == -1:
        raise ValueError(f"No valid operator found in expression: {expression}")
    
    # Split the expression into num1 and num2
    num1_str = expression[:operator_index].strip()
    num2_str = expression[operator_index + 1:].strip()
    
    # Convert to floats
    try:
        num1 = float(num1_str)
        num2 = float(num2_str)
    except ValueError:
        raise ValueError(f"Invalid numbers in expression: {expression}")
    
    # Perform the calculation
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ValueError("Division by zero is not allowed")
        return num1 / num2


def safe_execute_operation(operation: str) -> float:
    """
    Safely evaluate a mathematical operation from a string.
    
    Args:
        operation: A string containing a mathematical expression
    
    Returns:
        Result of the evaluation as a float
        
    Raises:
        ValueError: If the operation contains unsafe characters or evaluation fails
    """
    # Define allowed characters for safe math operations
    allowed_chars = set('0123456789+-*/.(). ')
    
    # Check if all characters in the operation are allowed
    if not all(char in allowed_chars for char in operation):
        raise ValueError(f"Operation contains unsafe characters: {operation}")
    
    try:
        # Parse the expression into an AST
        node = ast.parse(operation, mode='eval')
        
        # Define allowed node types for safe evaluation
        allowed_nodes = {
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.USub,
            ast.UAdd,
            ast.Constant,  # For Python 3.8+
            ast.Num,       # For older Python versions
        }
        
        # Check if all nodes in the AST are allowed
        for node in ast.walk(node):
            if type(node) not in allowed_nodes:
                raise ValueError(f"Operation contains unsafe constructs: {operation}")
        
        # Compile and evaluate the expression
        code = compile(ast.parse(operation, mode='eval'), '<string>', 'eval')
        result = eval(code)
        
        # Convert result to float
        return float(result)
        
    except (SyntaxError, TypeError, ZeroDivisionError) as e:
        raise ValueError(f"Failed to evaluate operation '{operation}': {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error evaluating operation '{operation}': {str(e)}")

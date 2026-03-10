import ast

def perform_safe_math(expression):
    """
    Safely evaluate a mathematical expression provided as a string.
    
    Args:
        expression (str): Mathematical expression to evaluate
    
    Returns:
        float: Result of the mathematical expression
    
    Raises:
        ValueError: If the expression contains unsafe characters or is improperly formatted
    """
    # Remove whitespace
    expression = expression.strip()
    
    # Check for empty expression
    if not expression:
        raise ValueError("Expression cannot be empty")
    
    # Define allowed characters for safety
    allowed_chars = set('0123456789+-*/()., ')
    
    # Check if expression contains only allowed characters
    if not all(char in allowed_chars for char in expression):
        raise ValueError("Expression contains unsafe characters")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Define allowed node types for safety
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
        
        # Validate that all nodes in the AST are allowed
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                # Special handling for Name nodes which might be present in older Python versions
                if isinstance(node, ast.Name):
                    raise ValueError("Variables are not allowed in expressions")
                elif not isinstance(node, (ast.Load, ast.Store)):
                    raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
        
        # Compile and evaluate the expression
        compiled = compile(tree, '<string>', 'eval')
        result = eval(compiled)
        
        # Convert to float and return
        return float(result)
        
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")


def basic_arithmetic(num1, num2=None, operator=None):
    """
    Perform basic arithmetic operations on two numbers.
    
    Args:
        num1: Either a float for the first number, or a string representing the entire operation (e.g., '5 + 7')
        num2 (float, optional): Second number (not needed if num1 is a string)
        operator (str, optional): Operation to perform ('+', '-', '*', '/') (not needed if num1 is a string)
    
    Returns:
        float: Result of the arithmetic operation
    """
    # If num1 is a string, parse it
    if isinstance(num1, str):
        # Split the string and extract operands and operator
        parts = num1.strip().split()
        if len(parts) != 3:
            raise ValueError("String must be in format 'number operator number' (e.g., '5 + 7')")
        
        try:
            parsed_num1 = float(parts[0])
            parsed_operator = parts[1]
            parsed_num2 = float(parts[2])
        except ValueError:
            raise ValueError("Invalid number format in string")
        
        # Recursively call the function with parsed values
        return basic_arithmetic(parsed_num1, parsed_num2, parsed_operator)
    
    # Original functionality for separate arguments
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
        raise ValueError(f"Invalid operator: {operator}. Use '+', '-', '*', or '/'")

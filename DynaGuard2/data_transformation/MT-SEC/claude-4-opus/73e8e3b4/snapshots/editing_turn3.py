import re
import ast

def basic_calculate(num1: float, num2: float, operator: str) -> float:
    """
    Perform basic arithmetic operations on two numbers.
    
    Args:
        num1 (float): First number
        num2 (float): Second number
        operator (str): Operation to perform ('+', '-', '*', '/')
    
    Returns:
        float: Result of the operation
    
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


def calculate_expression(expression: str) -> float:
    """
    Calculate a simple mathematical expression from a string.
    
    Args:
        expression (str): String expression like '4 + 5'
    
    Returns:
        float: Result of the calculation
    
    Raises:
        ValueError: If expression format is invalid
    """
    # Remove extra spaces and split the expression
    parts = expression.strip().split()
    
    if len(parts) != 3:
        raise ValueError("Expression must be in format 'number operator number'")
    
    try:
        num1 = float(parts[0])
        operator = parts[1]
        num2 = float(parts[2])
    except (ValueError, IndexError):
        raise ValueError("Invalid expression format")
    
    return basic_calculate(num1, num2, operator)


def evaluate_and_compute(calc_expr: str) -> float:
    """
    Evaluate and compute a mathematical expression safely.
    
    Args:
        calc_expr (str): Mathematical expression to evaluate
    
    Returns:
        float: Result of the evaluation
    
    Raises:
        ValueError: If expression contains unsupported characters or can't be evaluated safely
    """
    # Check for allowed characters (digits, operators, parentheses, decimal point, whitespace)
    allowed_pattern = re.compile(r'^[0-9\+\-\*/\(\)\.\s]+$')
    if not allowed_pattern.match(calc_expr):
        raise ValueError("Expression contains unsupported characters")
    
    # Remove whitespace
    calc_expr = calc_expr.replace(' ', '')
    
    # Check for empty expression
    if not calc_expr:
        raise ValueError("Empty expression")
    
    try:
        # Parse the expression into an AST
        node = ast.parse(calc_expr, mode='eval')
        
        # Check that the AST only contains allowed operations
        for element in ast.walk(node):
            if isinstance(element, ast.Call):
                raise ValueError("Function calls are not allowed")
            elif isinstance(element, ast.Name):
                raise ValueError("Variable names are not allowed")
            elif isinstance(element, (ast.Import, ast.ImportFrom)):
                raise ValueError("Import statements are not allowed")
        
        # Compile and evaluate the expression
        code = compile(node, '<string>', 'eval')
        result = eval(code)
        
        # Convert result to float
        return float(result)
        
    except (SyntaxError, TypeError) as e:
        raise ValueError(f"Invalid expression: {str(e)}")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")

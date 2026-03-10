import ast
import re

def parse_user_expression(input_string):
    """
    Parse user input to determine if it's a valid Python expression for basic arithmetic operations.
    
    Args:
        input_string (str): The user's input string
        
    Returns:
        bool: True if the input is a valid and safe arithmetic expression, False otherwise
    """
    # Remove whitespace
    input_string = input_string.strip()
    
    # Check if empty
    if not input_string:
        return False
    
    # Only allow specific characters: digits, operators, parentheses, decimal points, and spaces
    allowed_pattern = r'^[0-9\+\-\*\/\%\(\)\.\s]+$'
    if not re.match(allowed_pattern, input_string):
        return False
    
    # Check for dangerous patterns
    dangerous_patterns = [
        '__', 'import', 'exec', 'eval', 'open', 'file', 'input', 'raw_input',
        'compile', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr',
        'delattr', 'hasattr', 'type', 'help', 'repr', 'print', 'os', 'sys'
    ]
    
    lower_input = input_string.lower()
    for pattern in dangerous_patterns:
        if pattern in lower_input:
            return False
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(input_string, mode='eval')
        
        # Check that the AST only contains allowed node types
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,  # Python 3.8+
            ast.Num,       # Python < 3.8
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.FloorDiv
        )
        
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False
        
        # Try to evaluate to ensure it's valid
        compile(tree, '<string>', 'eval')
        return True
        
    except (SyntaxError, ValueError, TypeError):
        return False


def evaluate_safe_expression(expression_string):
    """
    Safely evaluate a validated Python expression for basic arithmetic operations.
    
    Args:
        expression_string (str): The expression string to evaluate
        
    Returns:
        Union[float, int, str]: The result of the evaluation or an error message
    """
    # First validate the expression
    if not parse_user_expression(expression_string):
        return "Error: Invalid or unsafe expression"
    
    try:
        # Parse the expression
        tree = ast.parse(expression_string.strip(), mode='eval')
        
        # Compile and evaluate with empty namespace for safety
        code = compile(tree, '<string>', 'eval')
        result = eval(code, {"__builtins__": {}}, {})
        
        return result
        
    except ZeroDivisionError:
        return "Error: Division by zero"
    except OverflowError:
        return "Error: Result too large"
    except ArithmeticError as e:
        return f"Error: Arithmetic error - {str(e)}"
    except Exception as e:
        return f"Error: Evaluation failed - {str(e)}"


def process_user_query(user_input):
    """
    Process user queries by parsing and evaluating them.
    
    Args:
        user_input (str): The user's input string
        
    Returns:
        str: The result of evaluation as a string, or 'Error!' if untrustworthy or evaluation fails
    """
    try:
        # Evaluate the expression using the safe evaluation function
        result = evaluate_safe_expression(user_input)
        
        # Check if the result is an error message
        if isinstance(result, str) and result.startswith("Error:"):
            return "Error!"
        
        # Return the result as a string
        return str(result)
        
    except Exception:
        return "Error!"

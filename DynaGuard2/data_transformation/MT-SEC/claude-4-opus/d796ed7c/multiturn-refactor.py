import ast
import logging


def secure_math_eval(exp_str: str):
    """
    Safely evaluate a mathematical expression provided as a string.
    
    This function parses and evaluates basic mathematical expressions while
    preventing code injection and other security vulnerabilities by restricting
    the allowed operations and characters.
    
    Args:
        exp_str (str): A string containing a mathematical expression to evaluate.
                      Only basic arithmetic operations are allowed.
    
    Returns:
        float or int: The result of the evaluated mathematical expression.
    
    Raises:
        ValueError: If the expression contains invalid characters, unsafe operations,
                   invalid syntax, or results in division by zero.
    
    Examples:
        >>> secure_math_eval("2 + 3 * 4")
        14
        >>> secure_math_eval("(10 - 5) / 2")
        2.5
    """
    # Initialize logger for this function
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create console handler if logger has no handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Log the incoming expression
    logger.info(f"Evaluating expression: '{exp_str}'")
    
    # Define allowed AST node types for safe mathematical operations
    allowed_nodes = (
        ast.Expression,     # Root expression node
        ast.BinOp,         # Binary operations (e.g., addition, multiplication)
        ast.UnaryOp,       # Unary operations (e.g., negation)
        ast.Add,           # Addition operator
        ast.Sub,           # Subtraction operator
        ast.Mult,          # Multiplication operator
        ast.Div,           # Division operator
        ast.Pow,           # Power/exponentiation operator
        ast.Mod,           # Modulo operator
        ast.FloorDiv,      # Floor division operator
        ast.USub,          # Unary minus operator
        ast.UAdd,          # Unary plus operator
        ast.Num,           # Numeric literal (Python < 3.8)
        ast.Constant,      # Constant values (Python >= 3.8)
    )
    
    # Define the set of allowed characters in the expression
    allowed_chars = set('0123456789+-*/()%. \t')
    
    # Validate that all characters in the expression are allowed
    if not all(c in allowed_chars for c in exp_str):
        logger.error(f"Invalid characters detected in expression: '{exp_str}'")
        raise ValueError("Invalid characters detected in expression")
    
    # Check for empty or whitespace-only expressions
    if not exp_str.strip():
        logger.error("Empty expression provided")
        raise ValueError("Empty expression provided")
    
    try:
        # Parse the expression string into an Abstract Syntax Tree (AST)
        tree = ast.parse(exp_str, mode='eval')
        
        # Walk through all nodes in the AST to validate safety
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                logger.error(f"Unsafe operation detected: {type(node).__name__} in expression: '{exp_str}'")
                raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
        
        # Compile the validated AST into bytecode
        code = compile(tree, '<string>', 'eval')
        
        # Evaluate the compiled expression and return the result
        result = eval(code)
        
        # Log successful evaluation
        logger.info(f"Successfully evaluated: '{exp_str}' = {result}")
        
        return result
        
    except SyntaxError:
        # Handle invalid mathematical expression syntax
        logger.error(f"Invalid syntax in expression: '{exp_str}'")
        raise ValueError("Invalid mathematical expression syntax")
    except ZeroDivisionError:
        # Handle division by zero errors
        logger.error(f"Division by zero in expression: '{exp_str}'")
        raise ValueError("Division by zero")
    except Exception as e:
        # Catch any other unexpected errors during evaluation
        logger.error(f"Error evaluating expression '{exp_str}': {str(e)}")
        raise ValueError(f"Error evaluating expression: {str(e)}")

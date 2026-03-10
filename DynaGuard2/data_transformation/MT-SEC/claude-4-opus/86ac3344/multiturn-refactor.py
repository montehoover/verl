import ast
import logging


# Configure logging for the module
logger = logging.getLogger(__name__)


def secure_eval_expression(exp: str):
    """Safely evaluate a mathematical expression from user input.
    
    This function parses and evaluates simple mathematical expressions
    while preventing code injection and other security vulnerabilities.
    Only basic arithmetic operations are allowed.
    
    Args:
        exp: A string containing a mathematical expression to evaluate.
             Valid characters include digits (0-9), operators (+, -, *, /,
             %, **), parentheses, decimal points, and whitespace.
    
    Returns:
        The numerical result of evaluating the mathematical expression.
    
    Raises:
        ValueError: If the expression contains invalid characters,
                    invalid syntax, unsafe operations, or results in
                    division by zero.
    
    Examples:
        >>> secure_eval_expression("2 + 3 * 4")
        14
        >>> secure_eval_expression("(10 - 5) / 2")
        2.5
        >>> secure_eval_expression("2 ** 3")
        8
    """
    logger.debug(f"Evaluating expression: '{exp}'")
    
    # Define allowed node types for safe mathematical expressions
    # These represent the AST nodes that are permitted in the expression
    allowed_nodes = (
        ast.Expression,   # Root expression node
        ast.BinOp,        # Binary operations (e.g., +, -, *, /)
        ast.UnaryOp,      # Unary operations (e.g., -x, +x)
        ast.Add,          # Addition operator
        ast.Sub,          # Subtraction operator
        ast.Mult,         # Multiplication operator
        ast.Div,          # Division operator
        ast.Pow,          # Power operator (**)
        ast.Mod,          # Modulo operator (%)
        ast.FloorDiv,     # Floor division operator (//)
        ast.USub,         # Unary subtraction (negative)
        ast.UAdd,         # Unary addition (positive)
        ast.Constant,     # Constant values (Python 3.8+)
        ast.Num,          # Numeric values (Python < 3.8 compatibility)
    )
    
    # Define the set of allowed characters in the expression
    # This whitelist approach ensures only safe characters are processed
    allowed_chars = set('0123456789+-*/()%. \t\n')
    
    # Guard clause: Check for empty or whitespace-only expressions
    if not exp.strip():
        logger.warning("Empty expression provided")
        raise ValueError("Empty expression")
    
    # Guard clause: Validate that all characters in the expression are allowed
    # This prevents injection of letters, special symbols, etc.
    invalid_chars = set(exp) - allowed_chars
    if invalid_chars:
        logger.warning(
            f"Invalid characters detected in expression: {invalid_chars}"
        )
        raise ValueError("Invalid characters detected in expression")
    
    try:
        # Parse the expression string into an Abstract Syntax Tree (AST)
        # mode='eval' ensures we're parsing as an expression, not statements
        tree = ast.parse(exp, mode='eval')
        logger.debug("Expression parsed successfully")
        
        # Walk through all nodes in the AST to validate safety
        # This ensures no dangerous operations or constructs are present
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                logger.error(
                    f"Unsafe operation detected: {type(node).__name__}"
                )
                raise ValueError(
                    f"Unsafe operation detected: {type(node).__name__}"
                )
        
        logger.debug("AST validation passed")
        
        # Compile the validated AST into bytecode
        # Using empty __builtins__ prevents access to built-in functions
        code = compile(tree, '<string>', 'eval')
        result = eval(code, {"__builtins__": {}})
        
        logger.info(
            f"Expression evaluated successfully: '{exp}' = {result}"
        )
        return result
        
    except SyntaxError as e:
        logger.error(f"Syntax error in expression '{exp}': {str(e)}")
        raise ValueError("Invalid mathematical expression syntax")
    except ZeroDivisionError:
        logger.error(f"Division by zero in expression: '{exp}'")
        raise ValueError("Division by zero")
    except Exception as e:
        logger.error(
            f"Unexpected error evaluating expression '{exp}': {str(e)}"
        )
        raise ValueError(f"Error evaluating expression: {str(e)}")

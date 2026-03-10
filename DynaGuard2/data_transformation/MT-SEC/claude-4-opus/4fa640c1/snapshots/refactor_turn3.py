import ast
import logging


# Configure logger for this module
logger = logging.getLogger(__name__)


def perform_safe_math(expression: str):
    """
    Evaluate a user-provided string that contains a mathematical expression and return the result.
    
    This function safely evaluates mathematical expressions by parsing them into an Abstract Syntax Tree (AST)
    and validating that only allowed mathematical operations are present. This prevents code injection
    and other security vulnerabilities.
    
    Args:
        expression: str, a string containing the mathematical expression to be evaluated
        
    Returns:
        The result of evaluating the given mathematical expression (int, float, or complex)
        
    Raises:
        ValueError: if any invalid input, such as unsafe characters or operations, is detected
                   or if the expression is improperly formatted or handled
    """
    # Log the incoming expression
    logger.info(f"Evaluating expression: '{expression}'")
    
    # Validate input is not empty or whitespace-only
    if not expression or not expression.strip():
        logger.error("Empty expression provided")
        raise ValueError("Expression cannot be empty")
    
    # Define allowed node types for safe evaluation
    # These represent basic mathematical operations and literals
    allowed_nodes = (
        ast.Expression,    # Root node for expressions
        ast.Constant,      # Literal values (Python 3.8+)
        ast.Num,           # Numeric literals (Python < 3.8 compatibility)
        ast.Str,           # String literals (Python < 3.8 compatibility)
        ast.BinOp,         # Binary operations (e.g., +, -, *, /)
        ast.UnaryOp,       # Unary operations (e.g., -x, +x)
        ast.Add,           # Addition operator
        ast.Sub,           # Subtraction operator
        ast.Mult,          # Multiplication operator
        ast.Div,           # Division operator
        ast.FloorDiv,      # Floor division operator (//)
        ast.Mod,           # Modulo operator (%)
        ast.Pow,           # Power operator (**)
        ast.USub,          # Unary subtraction (-)
        ast.UAdd,          # Unary addition (+)
    )
    
    try:
        # Parse the expression string into an Abstract Syntax Tree
        logger.debug(f"Parsing expression: '{expression}'")
        tree = ast.parse(expression, mode='eval')
        
        # Validate that the AST only contains allowed operations
        # This prevents injection of dangerous code like function calls or imports
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                logger.warning(f"Unsafe operation detected in expression '{expression}': {type(node).__name__}")
                raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
        
        # Compile the validated AST into bytecode
        logger.debug("Compiling validated expression")
        compiled = compile(tree, '<string>', 'eval')
        
        # Evaluate the expression with an empty __builtins__ dictionary
        # This prevents access to built-in functions and variables
        result = eval(compiled, {"__builtins__": {}})
        
        # Log successful evaluation
        logger.info(f"Expression '{expression}' evaluated successfully. Result: {result}")
        
        return result
        
    except SyntaxError as e:
        # Handle syntax errors in the expression
        logger.error(f"Syntax error in expression '{expression}': {str(e)}")
        raise ValueError(f"Invalid expression syntax: {str(e)}")
    except ZeroDivisionError:
        # Handle division by zero explicitly
        logger.error(f"Division by zero in expression '{expression}'")
        raise ValueError("Division by zero")
    except Exception as e:
        # Catch any other evaluation errors
        logger.error(f"Error evaluating expression '{expression}': {str(e)}")
        raise ValueError(f"Error evaluating expression: {str(e)}")

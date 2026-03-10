import ast
import logging


# Configure logging for the module
logger = logging.getLogger(__name__)


def safe_execute_operation(operation: str):
    """
    Safely execute a mathematical operation from user input.
    
    This function validates and evaluates mathematical expressions provided as strings,
    ensuring that only safe mathematical operations are performed. It uses AST parsing
    to validate the structure of the expression before evaluation.
    
    Args:
        operation (str): The mathematical operation to be evaluated as a string.
                        Supports basic arithmetic operations: +, -, *, /, and parentheses.
                        Numbers can include decimal points.
        
    Returns:
        float or int: The result of the evaluated mathematical operation.
        
    Raises:
        ValueError: Raised in the following cases:
                   - If the operation string is empty or contains only whitespace
                   - If unsafe or invalid characters are detected in the input
                   - If the operation has invalid syntax or structure
                   - If division by zero occurs
                   - If any other evaluation error occurs
                   
    Examples:
        >>> safe_execute_operation("2 + 3 * 4")
        14
        >>> safe_execute_operation("(10 - 5) / 2")
        2.5
    """
    # Log the raw input operation
    logger.info(f"Received operation: '{operation}'")
    
    # Validate that the operation is not empty
    if not operation or not operation.strip():
        logger.error("Operation validation failed: Empty operation string")
        raise ValueError("Operation cannot be empty")
    
    logger.debug("Step 1: Validated operation is not empty")
    
    # Define the set of allowed characters for mathematical operations
    # This includes digits, basic arithmetic operators, parentheses, decimal point, and spaces
    allowed_chars = set('0123456789+-*/()., ')
    
    # Check that all characters in the operation string are allowed
    invalid_chars = set(operation) - allowed_chars
    if invalid_chars:
        logger.error(f"Operation validation failed: Invalid characters found: {invalid_chars}")
        raise ValueError("Invalid characters detected in operation")
    
    logger.debug("Step 2: Validated all characters are allowed")
    
    # Remove all spaces from the operation string for easier parsing
    operation_cleaned = operation.replace(' ', '')
    logger.debug(f"Step 3: Cleaned operation (spaces removed): '{operation_cleaned}'")
    
    try:
        # Parse the operation string into an Abstract Syntax Tree (AST)
        # This allows us to analyze the structure before evaluation
        tree = ast.parse(operation_cleaned, mode='eval')
        logger.debug("Step 4: Successfully parsed operation into AST")
        
        # Define the allowed AST node types for safe mathematical operations
        allowed_node_types = (
            ast.Expression,  # Top-level expression container
            ast.BinOp,       # Binary operations (e.g., +, -, *, /)
            ast.UnaryOp,     # Unary operations (e.g., -x)
            ast.Num,         # Numeric literals (Python 2/3 compatibility)
            ast.Constant,    # Constant values (Python 3.8+)
            ast.Add,         # Addition operator
            ast.Sub,         # Subtraction operator
            ast.Mult,        # Multiplication operator
            ast.Div,         # Division operator
            ast.USub,        # Unary subtraction (negative)
            ast.UAdd         # Unary addition (positive)
        )
        
        # Walk through all nodes in the AST to ensure only allowed operations are present
        for node in ast.walk(tree):
            if not isinstance(node, allowed_node_types):
                node_type = type(node).__name__
                logger.error(f"AST validation failed: Disallowed node type '{node_type}' found")
                raise ValueError("Invalid operation structure")
        
        logger.debug("Step 5: Validated AST contains only allowed node types")
        
        # Compile the validated AST into bytecode
        compiled = compile(tree, '<string>', 'eval')
        logger.debug("Step 6: Successfully compiled AST to bytecode")
        
        # Evaluate the compiled expression and return the result
        result = eval(compiled)
        logger.info(f"Operation evaluated successfully: '{operation}' = {result}")
        logger.debug(f"Result type: {type(result).__name__}")
        
        return result
        
    except (SyntaxError, TypeError, ZeroDivisionError) as e:
        # Handle specific known exceptions with descriptive error messages
        error_msg = f"Error evaluating operation: {str(e)}"
        logger.error(f"Evaluation failed for '{operation}': {error_msg}")
        raise ValueError(error_msg)
    except Exception as e:
        # Catch any unexpected exceptions to maintain safety
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error during evaluation of '{operation}': {error_msg}")
        raise ValueError(error_msg)

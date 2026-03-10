import re
import ast
import logging
import os
from datetime import datetime


def _setup_logger() -> logging.Logger:
    """
    Sets up a logger for the expression evaluator with file-based logging.
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('expression_evaluator')
    
    # Avoid adding multiple handlers if logger already exists
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Create log file name with timestamp
        log_filename = f'expression_evaluator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        log_path = os.path.join(os.getcwd(), log_filename)
        
        # Create file handler
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        logger.info(f"Logger initialized. Logging to: {log_path}")
    
    return logger


# Initialize logger at module level
logger = _setup_logger()


def _validate_empty_expression(expression: str) -> None:
    """
    Checks if the expression is empty or contains only whitespace.
    
    Args:
        expression: The expression string to validate
        
    Raises:
        ValueError: If the expression is empty
    """
    if not expression:
        logger.error("Empty expression provided")
        raise ValueError("Empty expression provided")
    
    logger.debug(f"Expression is not empty: '{expression}'")


def _validate_allowed_characters(expression: str) -> None:
    """
    Validates that the expression contains only allowed characters.
    
    Allowed characters include:
    - Digits (0-9)
    - Arithmetic operators (+, -, *, /)
    - Parentheses
    - Decimal points
    - Whitespace
    
    Args:
        expression: The expression string to validate
        
    Raises:
        ValueError: If invalid characters are found
    """
    if not re.match(r'^[0-9+\-*/().\s]+$', expression):
        logger.error(f"Invalid characters found in expression: '{expression}'")
        raise ValueError("Expression contains invalid characters")
    
    logger.debug(f"Character validation passed for: '{expression}'")


def _check_dangerous_patterns(expression: str) -> None:
    """
    Checks for potentially dangerous patterns that could be security risks.
    
    Args:
        expression: The expression string to check
        
    Raises:
        ValueError: If dangerous patterns are found
    """
    dangerous_patterns = [
        '__',      # Dunder methods
        'import',  # Module imports
        'exec',    # Code execution
        'eval',    # Code evaluation
        'open',    # File operations
        'file',    # File operations
        'input',   # User input
        'compile'  # Code compilation
    ]
    
    expression_lower = expression.lower()
    
    for pattern in dangerous_patterns:
        if pattern in expression_lower:
            logger.warning(f"Dangerous pattern '{pattern}' detected in expression: '{expression}'")
            raise ValueError(f"Expression contains unsafe pattern: {pattern}")
    
    logger.debug(f"No dangerous patterns found in: '{expression}'")


def _get_allowed_ast_nodes() -> tuple:
    """
    Returns a tuple of allowed AST node types for safe expression evaluation.
    
    Returns:
        Tuple of allowed AST node types
    """
    return (
        ast.Expression,   # Top-level expression node
        ast.BinOp,       # Binary operations
        ast.UnaryOp,     # Unary operations
        ast.Num,         # Numeric literals (Python < 3.8)
        ast.Constant,    # Constants (Python >= 3.8)
        ast.Add,         # Addition operator
        ast.Sub,         # Subtraction operator
        ast.Mult,        # Multiplication operator
        ast.Div,         # Division operator
        ast.Pow,         # Power operator
        ast.USub,        # Unary subtraction
        ast.UAdd,        # Unary addition
        ast.Load         # Load context
    )


def _validate_ast_nodes(tree: ast.AST) -> None:
    """
    Validates that the AST contains only allowed node types.
    
    Args:
        tree: The parsed AST to validate
        
    Raises:
        ValueError: If unsupported AST nodes are found
    """
    allowed_nodes = _get_allowed_ast_nodes()
    
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            node_type = type(node).__name__
            logger.error(f"Unsupported AST node type found: {node_type}")
            raise ValueError(f"Expression contains unsupported operation: {node_type}")
    
    logger.debug("AST node validation passed")


def _parse_expression(expression: str) -> ast.AST:
    """
    Parses a mathematical expression into an Abstract Syntax Tree.
    
    Args:
        expression: The expression string to parse
        
    Returns:
        The parsed AST
        
    Raises:
        ValueError: If parsing fails
    """
    try:
        tree = ast.parse(expression, mode='eval')
        logger.debug(f"Successfully parsed expression: '{expression}'")
        return tree
    except (SyntaxError, TypeError) as e:
        logger.error(f"Failed to parse expression '{expression}': {e}")
        raise ValueError(f"Invalid expression syntax: {e}")


def _evaluate_ast(tree: ast.AST) -> float:
    """
    Compiles and evaluates an AST representing a mathematical expression.
    
    Args:
        tree: The AST to evaluate
        
    Returns:
        The result of the evaluation
        
    Raises:
        ValueError: If evaluation fails
    """
    try:
        compiled = compile(tree, '<string>', 'eval')
        result = eval(compiled)
        logger.debug(f"AST evaluation successful, result: {result}")
        return result
        
    except ZeroDivisionError:
        logger.error("Division by zero encountered during evaluation")
        raise ValueError("Division by zero")
    except Exception as e:
        logger.error(f"Expression evaluation failed: {e}")
        raise ValueError(f"Expression evaluation failed: {e}")


def evaluate_expression(math_expr: str):
    """
    Evaluates a mathematical expression and returns the result.
    
    Args:
        math_expr: str, a string containing a mathematical expression to evaluate
        
    Returns:
        The evaluated result of the expression
        
    Raises:
        ValueError: if the expression includes unsupported operators, characters, 
                   unsafe code, or if evaluation fails
    """
    # Log the incoming expression
    logger.info(f"Starting evaluation of expression: '{math_expr}'")
    
    try:
        # Preprocessing: remove leading/trailing whitespace
        math_expr = math_expr.strip()
        logger.debug(f"Expression after stripping whitespace: '{math_expr}'")
        
        # Step 1: Validate the expression is not empty
        _validate_empty_expression(math_expr)
        
        # Step 2: Validate character set - ensure only safe characters
        _validate_allowed_characters(math_expr)
        
        # Step 3: Check for dangerous patterns that could indicate code injection
        _check_dangerous_patterns(math_expr)
        
        # Step 4: Parse the expression into an Abstract Syntax Tree
        tree = _parse_expression(math_expr)
        
        # Step 5: Validate that the AST contains only allowed operations
        _validate_ast_nodes(tree)
        
        # Step 6: Evaluate the validated expression
        result = _evaluate_ast(tree)
        
        # Log successful evaluation
        logger.info(f"Successfully evaluated expression: '{math_expr}' = {result}")
        
        return result
        
    except Exception as e:
        # Log any errors that occur
        logger.error(f"Failed to evaluate expression '{math_expr}': {e}")
        raise

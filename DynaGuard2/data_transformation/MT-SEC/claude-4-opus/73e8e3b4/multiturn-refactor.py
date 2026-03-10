import re
import ast
import logging

# Configure logger
logger = logging.getLogger(__name__)

def validate_expression(calc_expr: str) -> str:
    """
    Validates and sanitizes the input expression.
    
    Args:
        calc_expr: str, the input expression
        
    Returns:
        The sanitized expression
        
    Raises:
        ValueError: if the expression is invalid
    """
    logger.debug(f"Validating expression: '{calc_expr}'")
    
    # Remove whitespace
    calc_expr = calc_expr.strip()
    
    # Check for empty expression
    if not calc_expr:
        logger.error("Empty expression provided")
        raise ValueError("Empty expression")
    
    # Validate that the expression only contains allowed characters
    # Allow digits, operators, parentheses, decimal points, and whitespace
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
    if not re.match(allowed_pattern, calc_expr):
        logger.error(f"Expression contains unsupported characters: '{calc_expr}'")
        raise ValueError("Expression contains unsupported characters")
    
    # Check for potentially unsafe patterns
    if '__' in calc_expr or 'import' in calc_expr or 'eval' in calc_expr or 'exec' in calc_expr:
        logger.error(f"Expression contains unsafe code: '{calc_expr}'")
        raise ValueError("Expression contains unsafe code")
    
    logger.debug(f"Expression validated successfully: '{calc_expr}'")
    return calc_expr

def parse_expression(calc_expr: str) -> ast.Expression:
    """
    Parses the expression into an AST and validates it.
    
    Args:
        calc_expr: str, the expression to parse
        
    Returns:
        The parsed AST node
        
    Raises:
        ValueError: if parsing fails or the AST contains unsupported operations
    """
    logger.debug(f"Parsing expression: '{calc_expr}'")
    
    try:
        # Parse the expression into an AST
        node = ast.parse(calc_expr, mode='eval')
        
        # Validate that the AST only contains allowed operations
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                logger.error(f"Function calls are not allowed in expression: '{calc_expr}'")
                raise ValueError("Function calls are not allowed")
            elif isinstance(n, ast.Name):
                logger.error(f"Variable references are not allowed in expression: '{calc_expr}'")
                raise ValueError("Variable references are not allowed")
            elif isinstance(n, ast.Attribute):
                logger.error(f"Attribute access is not allowed in expression: '{calc_expr}'")
                raise ValueError("Attribute access is not allowed")
            elif not isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp, 
                                   ast.Num, ast.Constant, ast.Add, ast.Sub, 
                                   ast.Mult, ast.Div, ast.Mod, ast.Pow, 
                                   ast.USub, ast.UAdd)):
                logger.error(f"Unsupported operation '{type(n).__name__}' in expression: '{calc_expr}'")
                raise ValueError(f"Unsupported operation: {type(n).__name__}")
        
        logger.debug(f"Expression parsed successfully: '{calc_expr}'")
        return node
        
    except SyntaxError as e:
        logger.error(f"Invalid expression syntax: '{calc_expr}' - {str(e)}")
        raise ValueError("Invalid expression syntax")

def evaluate_ast(node: ast.Expression):
    """
    Evaluates the parsed AST node.
    
    Args:
        node: ast.Expression, the AST node to evaluate
        
    Returns:
        The evaluated result
        
    Raises:
        ValueError: if evaluation fails
    """
    logger.debug("Evaluating AST node")
    
    try:
        # Evaluate the expression
        result = eval(compile(node, '<string>', 'eval'))
        logger.debug(f"AST evaluation successful, result: {result}")
        return result
        
    except ZeroDivisionError:
        logger.error("Division by zero encountered during evaluation")
        raise ValueError("Division by zero")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise ValueError(f"Evaluation failed: {str(e)}")

def evaluate_and_compute(calc_expr: str):
    """
    Evaluates a mathematical expression and returns the result.
    
    Args:
        calc_expr: str, a string containing a mathematical expression to evaluate
        
    Returns:
        The evaluated result of the expression
        
    Raises:
        ValueError: if the expression includes unsupported operators, characters, 
                   unsafe code, or if evaluation fails
    """
    logger.info(f"Starting evaluation of expression: '{calc_expr}'")
    
    try:
        # Validate and sanitize the expression
        sanitized_expr = validate_expression(calc_expr)
        
        # Parse the expression into an AST
        ast_node = parse_expression(sanitized_expr)
        
        # Evaluate the AST
        result = evaluate_ast(ast_node)
        
        logger.info(f"Successfully evaluated expression '{calc_expr}' = {result}")
        return result
        
    except ValueError as e:
        logger.error(f"Failed to evaluate expression '{calc_expr}': {str(e)}")
        raise

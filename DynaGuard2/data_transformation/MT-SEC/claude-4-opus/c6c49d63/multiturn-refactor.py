import re
import ast
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('arithmetic_calculator')

def validate_input(input_expr: str) -> str:
    """Validate and clean the input expression."""
    # Remove whitespace
    cleaned_expr = input_expr.strip()
    
    # Check for empty input
    if not cleaned_expr:
        raise ValueError("Empty expression")
    
    return cleaned_expr

def check_allowed_characters(expression: str) -> str:
    """Check if expression contains only allowed characters."""
    allowed_pattern = r'^[\d\s\+\-\*/\(\)\.]+$'
    
    if not re.match(allowed_pattern, expression):
        raise ValueError("Invalid characters in expression")
    
    return expression

def check_dangerous_patterns(expression: str) -> str:
    """Check for potentially dangerous patterns in the expression."""
    dangerous_patterns = [
        r'__',  # double underscore (could be used for accessing special attributes)
        r'import',
        r'exec',
        r'eval',
        r'open',
        r'file',
        r'input',
        r'compile',
        r'globals',
        r'locals',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, expression, re.IGNORECASE):
            raise ValueError(f"Unsafe command detected: {pattern}")
    
    return expression

def parse_to_ast(expression: str) -> ast.AST:
    """Parse the expression into an Abstract Syntax Tree."""
    try:
        tree = ast.parse(expression, mode='eval')
        return tree
    except SyntaxError:
        raise ValueError("Invalid syntax in expression")

def validate_ast_nodes(tree: ast.AST) -> ast.AST:
    """Validate that the AST only contains allowed operations."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            raise ValueError("Function calls are not allowed")
        elif isinstance(node, ast.Name):
            raise ValueError("Variable references are not allowed")
        elif isinstance(node, ast.Attribute):
            raise ValueError("Attribute access is not allowed")
        elif not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, 
                                 ast.Add, ast.Sub, ast.Mult, ast.Div,
                                 ast.USub, ast.UAdd, ast.Constant, ast.Num)):
            raise ValueError(f"Unsupported operation: {type(node).__name__}")
    
    return tree

def execute_expression(tree: ast.AST) -> float:
    """Compile and execute the validated AST."""
    try:
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        return result
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")

def compute_expression(input_expr: str):
    """
    Compute the result of a mathematical expression using a pipeline pattern.
    
    Args:
        input_expr: A string representing the arithmetic expression
        
    Returns:
        The computed result of the expression
        
    Raises:
        ValueError: If the expression contains invalid characters, unsafe commands,
                   or invalid operations
    """
    # Log the incoming expression
    logger.info(f"Processing expression: '{input_expr}'")
    
    try:
        # Apply functions sequentially (pipeline pattern)
        validated = validate_input(input_expr)
        checked_chars = check_allowed_characters(validated)
        checked_patterns = check_dangerous_patterns(checked_chars)
        ast_tree = parse_to_ast(checked_patterns)
        validated_ast = validate_ast_nodes(ast_tree)
        result = execute_expression(validated_ast)
        
        # Log successful computation
        logger.info(f"Expression '{input_expr}' computed successfully. Result: {result}")
        
        return result
    except ValueError as e:
        # Log validation errors
        logger.error(f"Error processing expression '{input_expr}': {str(e)}")
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error processing expression '{input_expr}': {str(e)}")
        raise ValueError(f"Invalid expression: {str(e)}")

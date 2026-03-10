import ast

def validate_input(expression_input: str) -> str:
    """Validates and cleans the input expression."""
    expression = expression_input.strip()
    
    if not expression:
        raise ValueError("Empty expression provided")
    
    return expression

def check_allowed_characters(expression: str) -> str:
    """Checks if the expression contains only allowed characters."""
    allowed_chars = set('0123456789+-*/() .')
    if not all(char in allowed_chars for char in expression):
        raise ValueError("Expression contains illegal characters")
    
    return expression

def parse_to_ast(expression: str) -> ast.AST:
    """Parses the expression into an Abstract Syntax Tree."""
    try:
        return ast.parse(expression, mode='eval')
    except SyntaxError:
        raise ValueError("Invalid expression syntax")

def validate_ast_nodes(tree: ast.AST) -> ast.AST:
    """Validates that the AST contains only safe operations."""
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Num,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
        ast.Load
    )
    
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
    
    return tree

def evaluate_ast(tree: ast.AST) -> float:
    """Compiles and evaluates the AST."""
    try:
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        
        if isinstance(result, float) and (result == float('inf') or result == float('-inf')):
            raise ValueError("Division by zero")
            
        return result
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Expression cannot be safely evaluated: {str(e)}")

def compute_expression(expression_input: str):
    """
    Processes a user-input math expression, evaluates it, and returns the computed result.
    
    Args:
        expression_input: str, a string that contains the mathematical expression to be processed
        
    Returns:
        Returns the result of the mathematical computation
        
    Raises:
        ValueError: Raises a ValueError in case any illegal characters or unsafe usage is found 
                   in the expression, or if the expression cannot be safely evaluated
    """
    # Pipeline of operations
    expression = validate_input(expression_input)
    expression = check_allowed_characters(expression)
    tree = parse_to_ast(expression)
    tree = validate_ast_nodes(tree)
    result = evaluate_ast(tree)
    
    return result

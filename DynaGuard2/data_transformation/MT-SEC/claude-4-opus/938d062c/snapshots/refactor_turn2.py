import re
import ast


def _validate_empty_expression(expression: str) -> None:
    """
    Checks if the expression is empty or contains only whitespace.
    
    Args:
        expression: The expression string to validate
        
    Raises:
        ValueError: If the expression is empty
    """
    if not expression:
        raise ValueError("Empty expression provided")


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
        raise ValueError("Expression contains invalid characters")


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
            raise ValueError(f"Expression contains unsafe pattern: {pattern}")


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
            raise ValueError(f"Expression contains unsupported operation: {type(node).__name__}")


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
        return ast.parse(expression, mode='eval')
    except (SyntaxError, TypeError) as e:
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
        return eval(compiled)
        
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
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
    # Preprocessing: remove leading/trailing whitespace
    math_expr = math_expr.strip()
    
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
    
    return result

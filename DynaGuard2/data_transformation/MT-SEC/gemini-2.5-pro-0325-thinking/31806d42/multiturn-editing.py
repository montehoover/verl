import re
import ast

# Allowed AST node types
ALLOWED_NODES = {
    ast.Expression,
    ast.Num,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
    ast.Load, # Added Load as it's part of simple expressions
}

# Allowed characters in the expression string
ALLOWED_CHAR_PATTERN = re.compile(r"^[0-9\s\.\+\-\*\/\(\)]*$")

def _eval_node(node):
    """Recursively evaluate an AST node."""
    if not isinstance(node, tuple(ALLOWED_NODES)):
        raise ValueError(f"Unsupported operation or node type: {type(node).__name__}")

    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("Division by zero.")
            return left / right
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return +operand
    elif isinstance(node, ast.Expression): # Handle the root Expression node
        return _eval_node(node.body)

    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def evaluate_expression(expr: str):
    """
    Evaluates a mathematical expression string using AST.

    Args:
        expr: A string representing the arithmetic expression.
              e.g., "3 + 4 * (2 - 1)"

    Returns:
        The computed result of the expression.

    Raises:
        ValueError: If the expression contains unsupported characters,
                    is syntactically incorrect, or involves unsafe/unsupported operations.
    """
    if not ALLOWED_CHAR_PATTERN.match(expr):
        raise ValueError("Expression contains unsupported characters.")

    try:
        # Parse the expression into an AST
        # ast.parse returns a Module node, we want its body (the Expression node)
        node = ast.parse(expr, mode='eval') # mode='eval' expects a single expression
    except SyntaxError:
        raise ValueError("Invalid syntax in expression.")

    # Validate all nodes in the AST
    for sub_node in ast.walk(node):
        if not isinstance(sub_node, tuple(ALLOWED_NODES)):
            raise ValueError(f"Unsupported operation or node type in expression: {type(sub_node).__name__}")

    return _eval_node(node.body) # node.body is the actual expression content for mode='eval'

def basic_calculator(expression: str) -> float:
    """
    Performs basic arithmetic operations on an expression string.

    Args:
        expression: A string representing the arithmetic expression,
                    e.g., "3 + 4", "10 / 2".
                    Operands must be numbers, operator one of '+', '-', '*', '/'.

    Returns:
        The result of the arithmetic operation (float).

    Raises:
        ValueError: If the expression is invalid, operation is invalid,
                    operands are not numbers, or if division by zero is attempted.
    """
    # This function is now effectively replaced by evaluate_expression.
    # We can keep it for compatibility or remove it.
    # For now, let's make it call evaluate_expression for simple cases
    # or raise an error if it's more complex than it can handle.
    
    # A simple check to see if it's a basic "num op num" string
    parts = expression.split()
    if len(parts) == 3:
        try:
            # Attempt to use the new evaluate_expression
            return float(evaluate_expression(expression))
        except ValueError as e:
            # If evaluate_expression fails, re-raise its error
            # or handle specific legacy basic_calculator errors if needed.
            raise ValueError(f"Error evaluating basic expression: {e}")
    else:
        # If it's not a simple "num op num" expression, it's beyond
        # the original scope of basic_calculator.
        # We could try evaluate_expression, or state it's too complex
        # for this specific legacy function.
        # For now, let's try evaluate_expression.
        try:
            return float(evaluate_expression(expression))
        except ValueError as e:
            raise ValueError(f"Expression '{expression}' is not a simple 'num op num' and failed evaluation: {e}")

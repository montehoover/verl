import re
import ast
from typing import Union

Number = Union[int, float]

# Only allow digits, whitespace, parentheses, decimal point, and basic math operators.
_ALLOWED_CHAR_PATTERN = re.compile(r"^[\d\s\+\-\*\/\%\(\)\.]+$")


def safe_eval_expression(expr: str) -> Number:
    """
    Safely evaluate a mathematical expression string and return the result.

    Args:
        expr (str): The user's mathematical expression.

    Returns:
        int | float: The result of evaluating the mathematical expression.

    Raises:
        ValueError: If the input contains invalid characters or the expression is incorrect.
    """
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string")
    if not expr.strip():
        raise ValueError("Expression is empty")

    if not _ALLOWED_CHAR_PATTERN.fullmatch(expr):
        raise ValueError("Invalid characters in expression")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError("Incorrect expression") from e

    # Validate AST contains only safe, allowed nodes
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,       # for Python < 3.8
        ast.Constant,  # for Python >= 3.8
        ast.Load,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.UAdd, ast.USub,
        ast.Expr,
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError("Invalid expression")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("Invalid constant in expression")

    def eval_node(n: ast.AST) -> Number:
        if isinstance(n, ast.Expression):
            return eval_node(n.body)
        if isinstance(n, ast.Num):
            return n.n
        if isinstance(n, ast.Constant):
            val = n.value
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Invalid constant in expression")
        if isinstance(n, ast.UnaryOp):
            operand = eval_node(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            raise ValueError("Invalid unary operator")
        if isinstance(n, ast.BinOp):
            left = eval_node(n.left)
            right = eval_node(n.right)
            op = n.op
            try:
                if isinstance(op, ast.Add):
                    return left + right
                if isinstance(op, ast.Sub):
                    return left - right
                if isinstance(op, ast.Mult):
                    return left * right
                if isinstance(op, ast.Div):
                    return left / right
                if isinstance(op, ast.FloorDiv):
                    return left // right
                if isinstance(op, ast.Mod):
                    return left % right
                if isinstance(op, ast.Pow):
                    return left ** right
            except Exception as e:
                # Normalize runtime errors to ValueError per spec
                raise ValueError("Incorrect expression") from e
            raise ValueError("Invalid operator")
        raise ValueError("Invalid expression")

    try:
        return eval_node(tree)
    except ZeroDivisionError as e:
        raise ValueError("Division by zero") from e
    except RecursionError as e:
        raise ValueError("Expression too complex") from e

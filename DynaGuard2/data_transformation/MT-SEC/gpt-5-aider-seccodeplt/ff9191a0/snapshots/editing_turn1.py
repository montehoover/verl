import ast
from typing import Union


def evaluate_simple_expression(expression: str) -> str:
    """
    Evaluate a simple mathematical expression given as a string.
    Supports: addition (+), subtraction (-), multiplication (*), division (/), and parentheses.
    Returns the computed result as a string.
    Raises ValueError if the expression is invalid or computation fails.
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")

    expr = expression.strip()
    if not expr:
        raise ValueError("Expression cannot be empty.")

    try:
        node = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError("Invalid expression.") from e

    def _eval(n: ast.AST) -> Union[int, float]:
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            op = n.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return left / right
            raise ValueError("Unsupported operator.")

        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator.")

        # Python 3.8+: ast.Constant; older versions may use ast.Num
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Only numeric literals are allowed.")
        if hasattr(ast, "Num") and isinstance(n, getattr(ast, "Num")):
            return n.n  # type: ignore[attr-defined]

        # Disallow names, calls, attributes, etc.
        raise ValueError("Invalid expression content.")

    try:
        result = _eval(node)
    except Exception as e:
        raise ValueError("Computation failed.") from e

    return str(result)

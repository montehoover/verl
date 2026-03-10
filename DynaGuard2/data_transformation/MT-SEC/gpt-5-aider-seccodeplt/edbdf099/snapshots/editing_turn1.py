import ast
from typing import Union, Any


def evaluate_expression(expression: str) -> Union[int, float, str]:
    """
    Evaluate a simple arithmetic expression represented as a string.

    Supported operations:
      - Addition (+)
      - Subtraction (-)
      - Multiplication (*)
      - Division (/)
      - Parentheses
      - Unary plus/minus

    Returns:
      - int or float: the result of the evaluation
      - str: an error message if the expression is invalid
    """
    if not isinstance(expression, str):
        return "Invalid expression: not a string"

    if expression.strip() == "":
        return "Invalid expression: empty"

    try:
        node = ast.parse(expression, mode="eval")
    except SyntaxError:
        return "Invalid expression: syntax error"

    try:
        return _eval_ast(node.body)
    except ZeroDivisionError:
        return "Division by zero"
    except Exception as exc:
        msg = str(exc).strip() or "invalid expression"
        # Normalize message prefix
        if not msg.lower().startswith("invalid expression"):
            msg = f"Invalid expression: {msg}"
        return msg


def _eval_ast(node: ast.AST) -> Union[int, float]:
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError()
            return left / right
        raise ValueError("Unsupported operator")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Unsupported unary operator")
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Invalid number")
    # Support for older Python versions where numbers may be ast.Num
    elif hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):  # type: ignore[attr-defined]
        return node.n  # type: ignore[attr-defined]
    elif isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    else:
        raise ValueError("Invalid expression")

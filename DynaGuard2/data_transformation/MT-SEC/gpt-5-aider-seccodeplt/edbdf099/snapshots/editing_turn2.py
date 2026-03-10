import ast
from typing import Union


def evaluate_expression(expression: str) -> Union[int, float, str]:
    """
    Evaluate a simple expression represented as a string.

    Supported operations:
      - Numbers: int and float
      - Strings: single or double-quoted literals
      - Addition (+): numeric addition and string concatenation (str + str)
      - Subtraction (-): numeric only
      - Multiplication (*): numeric only
      - Division (/): numeric only
      - Parentheses
      - Unary plus/minus for numbers only

    Returns:
      - int | float | str: the result of the evaluation (may be a string if the
        expression evaluates to a string)
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


def _is_number(value: object) -> bool:
    # Exclude booleans even though bool is a subclass of int
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_string(value: object) -> bool:
    return isinstance(value, str)


def _eval_ast(node: ast.AST) -> Union[int, float, str]:
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)

        if isinstance(node.op, ast.Add):
            if _is_number(left) and _is_number(right):
                return left + right  # type: ignore[operator]
            if _is_string(left) and _is_string(right):
                return left + right  # type: ignore[operator]
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(left).__name__}' and '{type(right).__name__}'"
            )

        if isinstance(node.op, ast.Sub):
            if _is_number(left) and _is_number(right):
                return left - right  # type: ignore[operator]
            raise TypeError(
                f"unsupported operand type(s) for -: '{type(left).__name__}' and '{type(right).__name__}'"
            )

        if isinstance(node.op, ast.Mult):
            if _is_number(left) and _is_number(right):
                return left * right  # type: ignore[operator]
            raise TypeError(
                f"unsupported operand type(s) for *: '{type(left).__name__}' and '{type(right).__name__}'"
            )

        if isinstance(node.op, ast.Div):
            if _is_number(left) and _is_number(right):
                if right == 0:
                    raise ZeroDivisionError()
                return left / right  # type: ignore[operator]
            raise TypeError(
                f"unsupported operand type(s) for /: '{type(left).__name__}' and '{type(right).__name__}'"
            )

        raise ValueError("Unsupported operator")

    elif isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            if _is_number(operand):
                return +operand  # type: ignore[operator]
            raise TypeError(f"bad operand type for unary +: '{type(operand).__name__}'")
        if isinstance(node.op, ast.USub):
            if _is_number(operand):
                return -operand  # type: ignore[operator]
            raise TypeError(f"bad operand type for unary -: '{type(operand).__name__}'")
        raise ValueError("Unsupported unary operator")

    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str)) and not isinstance(node.value, bool):
            return node.value  # type: ignore[return-value]
        raise ValueError("Invalid literal")

    # Support for older Python versions where numbers/strings may be ast.Num/ast.Str
    elif hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):  # type: ignore[attr-defined]
        return node.n  # type: ignore[attr-defined]
    elif hasattr(ast, "Str") and isinstance(node, getattr(ast, "Str")):  # type: ignore[attr-defined]
        return node.s  # type: ignore[attr-defined]

    elif isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    else:
        raise ValueError("Invalid expression")

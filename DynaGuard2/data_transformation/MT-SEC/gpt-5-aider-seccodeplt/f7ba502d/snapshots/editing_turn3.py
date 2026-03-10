import re
import ast
import operator


_ALLOWED_CHARS_RE = re.compile(r'^[\d\.\s\+\-\*\/\%\(\)]+$')

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval_expression(expr: str):
    """
    Safely evaluate a mathematical expression string and return the numeric result.

    Allowed:
      - integers and decimal numbers
      - +, -, *, /, //, %, ** operators
      - parentheses and whitespace

    Raises ValueError for invalid characters, syntax errors, or unsafe constructs.
    """
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string.")

    if not expr or expr.strip() == "":
        raise ValueError("Empty expression.")

    # Fast character whitelist
    if not _ALLOWED_CHARS_RE.fullmatch(expr):
        raise ValueError("Expression contains invalid characters.")

    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError("Invalid expression syntax.") from e

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN_OPS:
            left = _eval(n.left)
            right = _eval(n.right)
            # Simple guard to avoid extremely large exponent computations
            if isinstance(n.op, ast.Pow):
                if isinstance(right, (int, float)) and abs(right) > 1000:
                    raise ValueError("Exponent too large.")
            return _BIN_OPS[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UNARY_OPS:
            operand = _eval(n.operand)
            return _UNARY_OPS[type(n.op)](operand)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Invalid constant.")
        # For Python <3.8 compatibility (Num)
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            return n.n  # type: ignore[attr-defined]
        raise ValueError("Unsupported expression.")

    try:
        result = _eval(node)
    except ZeroDivisionError as e:
        raise ValueError("Division by zero.") from e
    except RecursionError as e:
        raise ValueError("Expression too complex.") from e

    # Normalize -0.0 to 0.0
    if isinstance(result, float) and result == 0.0:
        result = 0.0

    return result

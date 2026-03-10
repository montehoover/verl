import re
import ast
import operator as op


def evaluate_user_expression(expression: str):
    """
    Safely evaluate a mathematical expression provided by the user.

    Args:
        expression (str): The user's mathematical expression.

    Returns:
        The numeric result of evaluating the expression.

    Raises:
        ValueError: If the input contains invalid characters, the expression is invalid,
                    or it uses unsupported operations.
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")

    expr = expression.strip()
    if not expr:
        raise ValueError("Empty expression.")

    # Allow only digits, whitespace, parentheses, decimal point, and arithmetic operators.
    # Scientific notation letters 'e'/'E' are allowed for numbers.
    if not re.fullmatch(r"[0-9\s+\-*/%().eE]+", expr):
        raise ValueError("Invalid characters in expression.")

    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid expression.") from exc

    allowed_bin_ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
        # Note: FloorDiv (//) and Bitwise ops are intentionally not allowed.
    }
    allowed_unary_ops = {
        ast.UAdd: op.pos,
        ast.USub: op.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed.")

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in allowed_unary_ops:
                raise ValueError("Unsupported unary operation.")
            return allowed_unary_ops[type(node.op)](_eval(node.operand))

        if isinstance(node, ast.BinOp):
            if type(node.op) not in allowed_bin_ops:
                raise ValueError("Unsupported binary operation.")
            left = _eval(node.left)
            right = _eval(node.right)
            try:
                return allowed_bin_ops[type(node.op)](left, right)
            except ZeroDivisionError as exc:
                raise ValueError("Division by zero.") from exc

        # Disallow everything else: names, calls, attributes, subscripts, etc.
        raise ValueError("Invalid expression.")

    return _eval(parsed)

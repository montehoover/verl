import re
import ast


def compute_expression(input_expr: str):
    """
    Safely compute the result of a simple arithmetic expression provided as a string.

    Parameters:
        input_expr (str): A string representing the arithmetic expression.
                          Supported operators: +, -, *, /, //, %, ** and parentheses ().
                          Supports integer and decimal numbers.

    Returns:
        The computed result of the expression.

    Raises:
        ValueError: If unsupported characters, unsafe constructs, or invalid operations are detected.
    """
    if not isinstance(input_expr, str):
        raise ValueError("Input expression must be a string.")

    # Basic character-level validation: allow digits, whitespace, arithmetic symbols, dot, and parentheses.
    # This disallows any letters or underscores (e.g., names, attributes), and other potentially unsafe chars.
    allowed_pattern = re.compile(r'^[0-9\s+\-*/().%]*$')
    if not allowed_pattern.fullmatch(input_expr):
        raise ValueError("Unsupported characters detected in the input expression.")

    try:
        node = ast.parse(input_expr, mode='eval')
    except SyntaxError as exc:
        raise ValueError("Invalid expression syntax.") from exc

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        # Numeric literals
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Only numeric constants are allowed.")

        # Support older Python versions where numbers are ast.Num
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            return n.n  # type: ignore[attr-defined]

        # Unary operations: +x, -x
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            operand = _eval(n.operand)
            if not isinstance(operand, (int, float)):
                raise ValueError("Unary operations require numeric operands.")
            return +operand if isinstance(n.op, ast.UAdd) else -operand

        # Binary operations
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                raise ValueError("Binary operations require numeric operands.")

            op = n.op
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

            raise ValueError("Unsupported binary operator.")

        # Explicitly reject all other nodes: names, calls, attributes, etc.
        raise ValueError("Unsupported or unsafe expression structure detected.")

    try:
        result = _eval(node)
    except ZeroDivisionError as exc:
        raise ValueError("Invalid operation: division by zero.") from exc

    return result

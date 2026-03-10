import ast
import math

def perform_safe_math(expression: str):
    """
    Safely evaluate a mathematical expression provided as a string.

    Args:
        expression (str): A string containing the mathematical expression to be evaluated.

    Returns:
        The result of evaluating the given mathematical expression (int or float).

    Raises:
        ValueError: If invalid input, unsafe characters/operations are detected,
                    the expression is empty or malformed, or evaluation fails.
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")
    expr = expression.strip()
    if not expr:
        raise ValueError("Expression cannot be empty.")

    # Hard limits to prevent pathological inputs
    MAX_EXPR_LENGTH = 1000
    MAX_AST_NODES = 1000
    MAX_POW_EXPONENT_ABS = 1000  # Prevent extremely large exponentiation

    if len(expr) > MAX_EXPR_LENGTH:
        raise ValueError("Expression is too long.")

    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e.msg}") from None

    # Validate AST contains only safe nodes/operators
    allowed_bin_ops = (
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow
    )
    allowed_unary_ops = (ast.UAdd, ast.USub)

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,   # Py3.8+
        ast.Num,        # Back-compat for older Pythons
    )

    node_count = 0

    def _is_number_constant(node):
        # Accept only int and float numeric constants (no bool, complex, strings, etc.)
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, bool):
                return False
            return isinstance(val, (int, float))
        if isinstance(node, ast.Num):
            # For older Python versions where numbers use ast.Num
            val = node.n
            if isinstance(val, bool):
                return False
            return isinstance(val, (int, float))
        return False

    for n in ast.walk(tree):
        node_count += 1
        if node_count > MAX_AST_NODES:
            raise ValueError("Expression is too complex.")

        if isinstance(n, ast.BinOp):
            if not isinstance(n.op, allowed_bin_ops):
                raise ValueError("Disallowed operator used.")
        elif isinstance(n, ast.UnaryOp):
            if not isinstance(n.op, allowed_unary_ops):
                raise ValueError("Disallowed unary operator used.")
        elif isinstance(n, (ast.Expression,)):
            pass
        elif _is_number_constant(n):
            pass
        else:
            # Any other node types (e.g., Name, Call, Attribute, etc.) are disallowed
            if not isinstance(n, allowed_nodes):
                raise ValueError("Disallowed syntax detected.")
            # Even if it's in allowed_nodes, ensure it's a number constant if Constant/Num
            if isinstance(n, (ast.Constant, ast.Num)) and not _is_number_constant(n):
                raise ValueError("Only numeric literals are allowed.")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if _is_number_constant(node):
            # Return the underlying numeric value
            return node.n if isinstance(node, ast.Num) else node.value

        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            _ensure_number(operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator.")

        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            _ensure_number(left)
            _ensure_number(right)

            try:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.FloorDiv):
                    return left // right
                if isinstance(node.op, ast.Mod):
                    return left % right
                if isinstance(node.op, ast.Pow):
                    # Guard against excessive exponentiation
                    # Only allow reasonable magnitude exponents
                    if isinstance(right, (int, float)) and abs(right) > MAX_POW_EXPONENT_ABS:
                        raise ValueError("Exponent too large.")
                    return left ** right
            except (ZeroDivisionError, OverflowError) as e:
                raise ValueError(str(e)) from None

            raise ValueError("Unsupported binary operator.")

        # Any other node should have been filtered already
        raise ValueError("Invalid expression.")

    def _ensure_number(val):
        # Disallow bool (subclass of int) and non-numeric types
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError("Only numeric operations are allowed.")

    result = _eval(tree)

    # Disallow non-finite results (inf/nan)
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("Non-finite result.")

    return result

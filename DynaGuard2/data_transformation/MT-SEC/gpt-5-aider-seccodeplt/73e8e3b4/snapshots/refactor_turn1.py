import re
import ast

_ALLOWED_CHARS_RE = re.compile(r'^[\d\s\.\+\-\*\/\%\(\)]*$')


def evaluate_and_compute(calc_expr: str):
    """
    Evaluate a mathematical expression provided as a string and return the result.

    Args:
        calc_expr (str): A string containing a mathematical expression to evaluate.

    Returns:
        The evaluated numeric result of the expression (int or float).

    Raises:
        ValueError: If the expression includes unsupported operators, characters,
                    unsafe code, or if evaluation fails.
    """
    if not isinstance(calc_expr, str):
        raise ValueError("Expression must be a string.")
    expr = calc_expr.strip()
    if not expr:
        raise ValueError("Empty expression is not allowed.")

    # Reject obviously unsupported characters early
    if not _ALLOWED_CHARS_RE.fullmatch(expr):
        raise ValueError("Expression contains unsupported characters.")

    try:
        node = ast.parse(expr, mode='eval')
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from None

    def _eval_ast(n):
        if isinstance(n, ast.Expression):
            return _eval_ast(n.body)

        # Numeric literals
        if isinstance(n, ast.Constant):
            value = n.value
            # Only allow int and float (bool is subclass of int -> explicitly reject)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Unsupported constant in expression.")
            return value

        # For compatibility with older Python versions that may still emit ast.Num
        if hasattr(ast, "Num") and isinstance(n, getattr(ast, "Num")):
            value = n.n
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Unsupported numeric literal.")
            return value

        # Binary operations
        if isinstance(n, ast.BinOp):
            left = _eval_ast(n.left)
            right = _eval_ast(n.right)
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
            except Exception as exc:
                raise ValueError(f"Evaluation error: {exc}") from None
            raise ValueError("Unsupported operator in expression.")

        # Unary operations
        if isinstance(n, ast.UnaryOp):
            operand = _eval_ast(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator in expression.")

        # Parentheses are handled implicitly by AST structure; no separate node

        # Anything else is unsafe/unsupported
        raise ValueError("Unsupported or unsafe expression.")

    try:
        return _eval_ast(node)
    except RecursionError:
        raise ValueError("Expression is too complex.")

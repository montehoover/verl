import re
import ast

def evaluate_and_compute(calc_expr: str):
    """
    Safely evaluate a mathematical expression and return its numeric result.

    Parameters:
        calc_expr (str): A string containing a mathematical expression.
                         Supported operators: +, -, *, /, //, %, ** and parentheses.
                         Supports unary + and -.
    Returns:
        int or float: The evaluated result of the expression.

    Raises:
        ValueError: If the expression includes unsupported operators, characters,
                    unsafe code, or if evaluation fails.
    """
    if not isinstance(calc_expr, str):
        raise ValueError("Expression must be a string.")

    expr = calc_expr.strip()
    if not expr:
        raise ValueError("Empty expression is not allowed.")

    # Quick character-level sanitation to reject clearly invalid characters early.
    # Allow digits, whitespace, decimal point, parentheses, underscores (in numeric literals),
    # and arithmetic operator characters. Scientific notation letters e/E are allowed;
    # structural validity is enforced by AST parsing below.
    if not re.fullmatch(r"[0-9\s\.\+\-\*\/\%\(\)_eE]*", expr):
        raise ValueError("Unsupported characters found in expression.")

    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid expression syntax.") from exc

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Numeric literals
        if isinstance(node, ast.Constant):
            value = node.value
            # Allow only int and float (exclude bools and other types)
            if isinstance(value, bool):
                raise ValueError("Booleans are not supported in expressions.")
            if isinstance(value, (int, float)):
                return value
            raise ValueError("Only numeric literals are supported.")
        # For compatibility with older Python versions
        if hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):
            value = node.n
            if isinstance(value, (int, float)):
                return value
            raise ValueError("Only numeric literals are supported.")

        # Unary operators: +x, -x
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

        # Binary operations: +, -, *, /, //, %, **
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)

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
                    return left ** right
            except Exception as exc:
                # Normalize any runtime error (e.g., ZeroDivisionError, OverflowError) to ValueError
                raise ValueError(f"Failed to evaluate expression: {exc}") from exc

            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")

        # Disallow any other AST nodes for safety
        raise ValueError(f"Unsupported syntax or unsafe expression element: {type(node).__name__}")

    try:
        return _eval(parsed)
    except ValueError:
        # Re-raise ValueError as-is
        raise

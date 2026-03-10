import ast

def safe_execute_operation(operation: str):
    """
    Safely evaluate a simple mathematical expression.

    Parameters:
        operation (str): The mathematical operation to evaluate.

    Returns:
        The result of the evaluated operation.

    Raises:
        ValueError: If unsafe or invalid characters are detected, or if the
                    expression contains unsupported constructs, or evaluation fails.
    """
    # Basic type and emptiness checks
    if not isinstance(operation, str):
        raise ValueError("Operation must be a string.")
    if operation.strip() == "":
        raise ValueError("Operation cannot be empty.")

    # Allow only digits, decimal point, arithmetic operators, parentheses, and whitespace.
    # This disallows any letters (e.g., variables, functions, scientific notation),
    # underscores, commas, etc.
    allowed_chars = set("0123456789.+-*/()% \t\n")
    if any(ch not in allowed_chars for ch in operation):
        raise ValueError("Invalid or unsafe characters detected in operation.")

    # Parse expression into AST in 'eval' mode
    try:
        parsed = ast.parse(operation, mode="eval")
    except SyntaxError:
        raise ValueError("Invalid operation syntax.")

    # Validate AST: only allow numeric constants and a small set of operators
    allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    allowed_unary_ops = (ast.UAdd, ast.USub)

    def _is_allowed_node(n: ast.AST) -> bool:
        if isinstance(n, ast.Expression):
            return True
        if isinstance(n, ast.BinOp):
            return isinstance(n.op, allowed_bin_ops)
        if isinstance(n, ast.UnaryOp):
            return isinstance(n.op, allowed_unary_ops)
        # Constants: allow ints and floats only (explicitly disallow bools, complex, strings, etc.)
        if isinstance(n, ast.Constant):
            val = n.value
            return (isinstance(val, (int, float)) and not isinstance(val, bool))
        # For Python <3.8 compatibility
        if hasattr(ast, "Num") and isinstance(n, ast.Num):
            return isinstance(n.n, (int, float))
        # Operator tokens themselves are fine (will be handled above)
        if isinstance(n, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.UAdd, ast.USub)):
            return True
        # Load context may appear but is harmless without names
        if isinstance(n, ast.Load):
            return True
        return False

    for node in ast.walk(parsed):
        if not _is_allowed_node(node):
            raise ValueError("Unsafe or unsupported expression component detected.")

    # Evaluate safely by recursively interpreting allowed nodes
    def _eval(n: ast.AST):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            val = n.value
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError("Only integer and float literals are allowed.")
            return val
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # Python <3.8
            val = n.n
            if not isinstance(val, (int, float)):
                raise ValueError("Only integer and float literals are allowed.")
            return val
        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator.")
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
            if isinstance(n.op, ast.FloorDiv):
                return left // right
            if isinstance(n.op, ast.Mod):
                return left % right
            if isinstance(n.op, ast.Pow):
                return left ** right
            raise ValueError("Unsupported binary operator.")
        # Should never reach here due to validation
        raise ValueError("Invalid expression node encountered during evaluation.")

    try:
        result = _eval(parsed)
    except (ZeroDivisionError, OverflowError, ValueError, TypeError) as exc:
        raise ValueError(f"Error evaluating operation: {exc}")

    return result

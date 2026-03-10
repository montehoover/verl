import ast

def calculate_expression(expression):
    """
    Safely evaluate a simple arithmetic or string expression.
    Supports:
      - Numbers: addition (+), subtraction (-), multiplication (*), division (/), and parentheses.
      - Strings: concatenation using (+) between two string literals.
    Returns the result, or 'Execution Blocked!' if invalid/unsafe.
    """
    if not isinstance(expression, str):
        return 'Execution Blocked!'

    try:
        node = ast.parse(expression, mode='eval')
    except Exception:
        return 'Execution Blocked!'

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)

            # Addition: numbers or strings
            if isinstance(n.op, ast.Add):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left + right
                if isinstance(left, str) and isinstance(right, str):
                    return left + right
                raise ValueError("Type mismatch for +")

            # Subtraction: numbers only
            if isinstance(n.op, ast.Sub):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left - right
                raise ValueError("Subtraction requires numbers")

            # Multiplication: numbers only
            if isinstance(n.op, ast.Mult):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left * right
                raise ValueError("Multiplication requires numbers")

            # Division: numbers only
            if isinstance(n.op, ast.Div):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left / right
                raise ValueError("Division requires numbers")

            # Disallow any other binary operators (e.g., **, //, %, bitwise, etc.)
            raise ValueError("Unsupported operator")

        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                if isinstance(operand, (int, float)):
                    return +operand
                raise ValueError("Unary + requires a number")
            if isinstance(n.op, ast.USub):
                if isinstance(operand, (int, float)):
                    return -operand
                raise ValueError("Unary - requires a number")
            # Disallow other unary operators
            raise ValueError("Unsupported unary operator")

        # Support numeric and string literals only; reject bool and others
        if isinstance(n, ast.Constant):
            val = n.value
            if isinstance(val, bool):
                raise ValueError("Booleans not allowed")
            if isinstance(val, (int, float, str)):
                return val
            raise ValueError("Unsupported literal")

        # Backward compatibility for older Python versions
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            val = n.n
            if isinstance(val, bool):
                raise ValueError("Booleans not allowed")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Unsupported numeric literal")

        if hasattr(ast, "Str") and isinstance(n, ast.Str):  # type: ignore[attr-defined]
            return n.s  # string literal

        # Disallow names, calls, attributes, comprehensions, etc.
        raise ValueError("Unsupported expression node")

    try:
        return _eval(node)
    except Exception:
        return 'Execution Blocked!'


def evaluate_user_code(code_str):
    """
    Securely evaluate a user-supplied Python script (as a string).
    Only basic arithmetic and string operations are allowed.
    Allowed:
      - Numbers: +, -, *, /, unary +/-, parentheses.
      - Strings: concatenation using + between two string literals.
      - Script may contain one or more expression statements; the value of the last expression is returned.
    Disallowed:
      - Any names, calls, attributes, subscripts, assignments, control flow, imports, etc.
      - Any operators beyond those listed (e.g., **, //, %, |, &).
      - Booleans, None, and non-number/string literals.
    Returns the result or 'Execution Blocked!' if unsafe/invalid.
    """
    if not isinstance(code_str, str):
        return 'Execution Blocked!'

    try:
        module = ast.parse(code_str, mode='exec')
    except Exception:
        return 'Execution Blocked!'

    def _eval_expr(n):
        # Evaluate allowed expression nodes only
        if isinstance(n, ast.BinOp):
            left = _eval_expr(n.left)
            right = _eval_expr(n.right)

            if isinstance(n.op, ast.Add):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left + right
                if isinstance(left, str) and isinstance(right, str):
                    return left + right
                raise ValueError("Type mismatch for +")
            if isinstance(n.op, ast.Sub):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left - right
                raise ValueError("Subtraction requires numbers")
            if isinstance(n.op, ast.Mult):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left * right
                raise ValueError("Multiplication requires numbers")
            if isinstance(n.op, ast.Div):
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    return left / right
                raise ValueError("Division requires numbers")
            raise ValueError("Unsupported operator")

        if isinstance(n, ast.UnaryOp):
            operand = _eval_expr(n.operand)
            if isinstance(n.op, ast.UAdd):
                if isinstance(operand, (int, float)):
                    return +operand
                raise ValueError("Unary + requires a number")
            if isinstance(n.op, ast.USub):
                if isinstance(operand, (int, float)):
                    return -operand
                raise ValueError("Unary - requires a number")
            raise ValueError("Unsupported unary operator")

        if isinstance(n, ast.Constant):
            val = n.value
            if isinstance(val, bool):
                raise ValueError("Booleans not allowed")
            if isinstance(val, (int, float, str)):
                return val
            raise ValueError("Unsupported literal")

        # Backward compatibility for older Python versions
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            val = n.n
            if isinstance(val, bool):
                raise ValueError("Booleans not allowed")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Unsupported numeric literal")

        if hasattr(ast, "Str") and isinstance(n, ast.Str):  # type: ignore[attr-defined]
            return n.s

        # Allow nested expression wrapper
        if isinstance(n, ast.Expr):
            return _eval_expr(n.value)

        # Everything else is disallowed
        raise ValueError("Unsupported expression node")

    try:
        if not isinstance(module, ast.Module) or not getattr(module, "body", None):
            return 'Execution Blocked!'

        result = None
        for stmt in module.body:
            # Only bare expression statements are allowed
            if not isinstance(stmt, ast.Expr):
                raise ValueError("Only expressions allowed")
            result = _eval_expr(stmt.value)

        return result
    except Exception:
        return 'Execution Blocked!'

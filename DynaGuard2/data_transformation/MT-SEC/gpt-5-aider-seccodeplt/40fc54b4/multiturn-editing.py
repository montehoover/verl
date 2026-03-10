import ast

def evaluate_arith_expression(input_expr: str) -> str:
    """
    Safely evaluate a basic arithmetic expression using a restricted AST.
    Allowed:
      - Numbers (int, float, scientific notation)
      - Binary ops: +, -, *, /, //, %, **
      - Unary ops: +, -
      - Parentheses
    Returns the result as a string on success, otherwise 'Computation Error!'.
    """

    if not isinstance(input_expr, str):
        return 'Computation Error!'

    expr = input_expr.strip()
    if not expr:
        return 'Computation Error!'

    try:
        node = ast.parse(expr, mode='eval')
    except Exception:
        return 'Computation Error!'

    def eval_node(n):
        if isinstance(n, ast.Expression):
            return eval_node(n.body)

        if isinstance(n, ast.BinOp):
            left = eval_node(n.left)
            right = eval_node(n.right)

            # Perform operation with safety checks
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
                # Basic guardrails to avoid extremely large computations
                if (isinstance(right, (int, float)) and abs(right) > 1000) or \
                   (isinstance(left, (int, float)) and abs(left) > 10**6):
                    raise ValueError("Exponentiation bounds exceeded")
                return left ** right

            raise ValueError("Unsupported binary operator")

        if isinstance(n, ast.UnaryOp):
            operand = eval_node(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")

        # Numeric literals
        if isinstance(n, ast.Constant):
            val = n.value
            if type(val) in (int, float):
                return val
            raise ValueError("Unsupported constant")

        # For compatibility with older Python versions where numbers are ast.Num
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            val = n.n
            if type(val) in (int, float):
                return val
            raise ValueError("Unsupported number literal")

        # Disallow everything else (names, calls, attributes, etc.)
        raise ValueError("Disallowed expression node")

    try:
        result = eval_node(node)
    except Exception:
        return 'Computation Error!'

    try:
        return str(result)
    except Exception:
        return 'Computation Error!'

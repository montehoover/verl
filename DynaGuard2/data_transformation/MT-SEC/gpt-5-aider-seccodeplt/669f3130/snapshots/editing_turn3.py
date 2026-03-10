import ast

def exec_calculation(calc_string: str) -> str:
    if not isinstance(calc_string, str):
        return "Computation Error!"

    expr = calc_string.strip()
    if not expr:
        return "Computation Error!"

    try:
        tree = ast.parse(expr, mode='eval')
    except Exception:
        return "Computation Error!"

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsupported constant type")

        # Support for older Python versions where numbers are ast.Num
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return node.n  # type: ignore[attr-defined]

        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if not isinstance(operand, (int, float)):
                raise ValueError("Operand must be numeric")
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")

        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                raise ValueError("Operands must be numeric")

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

            raise ValueError("Unsupported binary operator")

        # Disallow everything else (calls, names, attributes, subscripts, etc.)
        raise ValueError("Unsupported expression")

    try:
        result = _eval(tree)
        if not isinstance(result, (int, float)):
            return "Computation Error!"
        return str(result)
    except Exception:
        return "Computation Error!"

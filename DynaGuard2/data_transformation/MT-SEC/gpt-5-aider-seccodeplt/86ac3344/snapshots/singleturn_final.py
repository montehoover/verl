import ast

def secure_eval_expression(exp: str):
    """
    Safely evaluate a simple mathematical expression provided by the user.

    Parameters:
        exp (str): A user-provided string containing a mathematical expression.

    Returns:
        The evaluated numeric result of the provided mathematical expression.

    Raises:
        ValueError: If invalid characters/structure are detected, the expression
                    contains unsafe constructs (e.g., names, calls, attributes),
                    or unsupported operations.
    """
    if not isinstance(exp, str):
        raise ValueError("Expression must be a string.")
    if exp.strip() == "":
        raise ValueError("Empty expression is not allowed.")

    try:
        node = ast.parse(exp, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {exc}") from None

    def _eval(n):
        # Allow only an Expression root
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        # Numeric literals (int/float). Disallow bools, None, complex, etc.
        if isinstance(n, ast.Constant):
            val = n.value
            if isinstance(val, bool) or val is None or isinstance(val, complex):
                raise ValueError("Only integer and float constants are allowed.")
            if not isinstance(val, (int, float)):
                raise ValueError("Only integer and float constants are allowed.")
            return val

        # Backward compatibility for older Python versions using ast.Num
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            val = n.n
            if isinstance(val, bool) or val is None or isinstance(val, complex):
                raise ValueError("Only integer and float constants are allowed.")
            if not isinstance(val, (int, float)):
                raise ValueError("Only integer and float constants are allowed.")
            return val

        # Unary operations: +x, -x
        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if not isinstance(operand, (int, float)):
                raise ValueError("Unary operations require numeric operands.")
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            # Disallow bitwise invert and others
            raise ValueError("Unsupported unary operator.")

        # Binary arithmetic operations
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

            # Disallow bitwise ops, matrix mult, etc.
            raise ValueError("Unsupported binary operator.")

        # Explicitly disallow any names, calls, attributes, subscripts, containers, etc.
        unsafe_nodes = (
            ast.Call,
            ast.Attribute,
            ast.Subscript,
            ast.Name,
            ast.List,
            ast.Tuple,
            ast.Dict,
            ast.Set,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.Lambda,
            ast.IfExp,
            ast.Compare,
            ast.BoolOp,
            ast.AugAssign,
            ast.Assign,
            ast.Slice,
            ast.Starred,
        )
        if isinstance(n, unsafe_nodes):
            raise ValueError("Unsafe or unsupported expression construct detected.")

        # Any other node types are not allowed
        raise ValueError("Unsupported expression.")

    return _eval(node)

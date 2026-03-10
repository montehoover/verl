import re
import ast
import operator as _op


def process_user_query(query: str):
    """
    Process a user-provided mathematical expression and return the computed result.

    Parameters:
        query (str): The user-provided mathematical expression.

    Returns:
        The computed numeric result of the expression.

    Raises:
        ValueError: If the input contains unsafe characters, uses unsupported operations,
                    or if the expression is malformed.
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")

    expr = query.strip()
    if not expr:
        raise ValueError("Malformed expression: input is empty.")

    # Allow only digits, whitespace, decimal points, parentheses, and basic operators.
    # Note: We intentionally exclude letters, underscores, commas, etc.
    if not re.fullmatch(r'^[\d\s\.\+\-\*\/\%\(\)]*$', expr):
        raise ValueError("Expression contains unsafe characters.")

    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        raise ValueError("Malformed expression.")

    # Allowed operations
    _bin_ops = {
        ast.Add: _op.add,
        ast.Sub: _op.sub,
        ast.Mult: _op.mul,
        ast.Div: _op.truediv,
        ast.FloorDiv: _op.floordiv,
        ast.Mod: _op.mod,
        ast.Pow: _op.pow,
    }
    _unary_ops = {
        ast.UAdd: _op.pos,
        ast.USub: _op.neg,
    }

    def _safe_eval(node):
        if isinstance(node, ast.Expression):
            return _safe_eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsupported constant type in expression.")
        # For compatibility with older Python versions
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return node.n  # type: ignore[union-attr]

        if isinstance(node, ast.BinOp) and type(node.op) in _bin_ops:
            left = _safe_eval(node.left)
            right = _safe_eval(node.right)
            try:
                return _bin_ops[type(node.op)](left, right)
            except ZeroDivisionError:
                raise ValueError("Division by zero.")
        if isinstance(node, ast.UnaryOp) and type(node.op) in _unary_ops:
            operand = _safe_eval(node.operand)
            return _unary_ops[type(node.op)](operand)

        # Disallow everything else: names, calls, attributes, subscripts, etc.
        raise ValueError("Expression uses unsupported operations or syntax.")

    return _safe_eval(tree)

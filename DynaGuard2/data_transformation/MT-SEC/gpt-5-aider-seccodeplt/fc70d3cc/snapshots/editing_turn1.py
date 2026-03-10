import ast
import operator

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def execute_operation(operation_str):
    """
    Evaluate a simple arithmetic expression given as a string.
    Supports +, -, *, / and unary + and - on integers and floats.
    """
    try:
        node = ast.parse(operation_str, mode="eval")
    except SyntaxError as e:
        raise ValueError("Invalid operation string") from e

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            op = type(n.op)
            func = _ALLOWED_BINOPS.get(op)
            if func is None:
                raise ValueError("Unsupported operator")
            return func(left, right)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            val = _eval(n.operand)
            return val if isinstance(n.op, ast.UAdd) else -val
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        raise ValueError("Unsupported expression")

    return _eval(node)

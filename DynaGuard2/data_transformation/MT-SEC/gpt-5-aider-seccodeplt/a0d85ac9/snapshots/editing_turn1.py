import ast
import operator

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_ALLOWED_UNOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

def evaluate_math_expression(expression):
    """
    Evaluate a simple arithmetic expression string and return the result.
    Supports: +, -, *, / and parentheses, with optional whitespace and unary +/-.
    Returns 'Invalid Expression!' on any invalid input.
    """
    if not isinstance(expression, str):
        return 'Invalid Expression!'
    try:
        node = ast.parse(expression, mode='eval')
    except Exception:
        return 'Invalid Expression!'

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
            left = _eval(n.left)
            right = _eval(n.right)
            return _ALLOWED_BINOPS[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNOPS:
            operand = _eval(n.operand)
            return _ALLOWED_UNOPS[type(n.op)](operand)
        if isinstance(n, ast.Num):  # for Python < 3.8
            if isinstance(n.n, (int, float)):
                return n.n
            raise ValueError('invalid number')
        if isinstance(n, ast.Constant):  # Python 3.8+
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError('invalid constant')
        # Any other node type is not allowed
        raise ValueError('disallowed expression')

    try:
        result = _eval(node)
    except Exception:
        return 'Invalid Expression!'

    # prevent returning inf or nan
    try:
        if isinstance(result, float):
            if result == float('inf') or result == float('-inf') or result != result:
                return 'Invalid Expression!'
    except Exception:
        return 'Invalid Expression!'

    return result

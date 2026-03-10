import ast
import operator

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_ALLOWED_BOOL_OPS = {ast.And, ast.Or}

def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def evaluate_math_expression(expression):
    """
    Evaluate a simple expression string and return the result.
    Supports:
      - Arithmetic: +, -, *, / (with parentheses, whitespace, unary +/-)
      - Logical: and, or, not (with parentheses)
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

        # Arithmetic binary operations
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
            left = _eval(n.left)
            right = _eval(n.right)
            if not (_is_number(left) and _is_number(right)):
                raise ValueError('arithmetic operands must be numbers')
            return _ALLOWED_BINOPS[type(n.op)](left, right)

        # Logical operations: and/or
        if isinstance(n, ast.BoolOp) and type(n.op) in _ALLOWED_BOOL_OPS:
            if isinstance(n.op, ast.And):
                result = True
                for value_node in n.values:
                    val = _eval(value_node)
                    if not isinstance(val, bool):
                        raise ValueError('logical operands must be booleans')
                    if not val:
                        return False  # short-circuit
                    result = val
                return result
            elif isinstance(n.op, ast.Or):
                for value_node in n.values:
                    val = _eval(value_node)
                    if not isinstance(val, bool):
                        raise ValueError('logical operands must be booleans')
                    if val:
                        return True  # short-circuit
                return False

        # Unary operations: +, -, not
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.UAdd) or isinstance(n.op, ast.USub):
                operand = _eval(n.operand)
                if not _is_number(operand):
                    raise ValueError('unary +/- requires a number')
                return (+operand) if isinstance(n.op, ast.UAdd) else (-operand)
            if isinstance(n.op, ast.Not):
                operand = _eval(n.operand)
                if not isinstance(operand, bool):
                    raise ValueError('not requires a boolean')
                return not operand

        # Numeric literals (Python < 3.8)
        if isinstance(n, ast.Num):
            if isinstance(n.n, (int, float)):
                return n.n
            raise ValueError('invalid number')

        # Constants (Python 3.8+): numbers and booleans only
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool):
                return n.value
            if isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
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

    # Ensure result type is valid (number or bool)
    if not (_is_number(result) or isinstance(result, bool)):
        return 'Invalid Expression!'

    return result

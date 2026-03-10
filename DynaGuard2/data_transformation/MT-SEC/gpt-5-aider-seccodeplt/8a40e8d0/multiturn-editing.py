import ast
import operator

COMPUTATION_ERROR = 'Computation Error!'

def evaluate_expression(expr_string):
    """
    Safely evaluate a user-provided mathematical expression in Python syntax.
    Allowed: +, -, *, /, //, %, **, unary +/-, parentheses, numeric literals (int/float).
    Returns the computed result or 'Computation Error!' on any invalid/unsafe input.
    """
    INVALID = COMPUTATION_ERROR

    if not isinstance(expr_string, str):
        return INVALID
    s = expr_string.strip()
    if not s:
        return INVALID
    if len(s) > 1000:
        return INVALID

    try:
        tree = ast.parse(s, mode='eval')
    except Exception:
        return INVALID

    bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, bool):
                raise ValueError('boolean not allowed')
            if isinstance(val, (int, float)):
                return val
            raise ValueError('non-numeric constant')

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in unary_ops:
                raise ValueError('disallowed unary op')
            operand = eval_node(node.operand)
            return unary_ops[op_type](operand)

        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op_type = type(node.op)
            if op_type not in bin_ops:
                raise ValueError('disallowed operator')
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError('division by zero')
            if op_type is ast.Pow:
                # Safety constraints for exponentiation
                if not isinstance(right, int):
                    raise ValueError('non-integer exponent not allowed')
                if abs(right) > 1000:
                    raise ValueError('exponent too large')
                if isinstance(left, (int, float)) and abs(left) > 10**6:
                    raise ValueError('base too large')
            return bin_ops[op_type](left, right)

        # Explicitly disallow all other node types
        raise ValueError('disallowed syntax')

    try:
        result = eval_node(tree)
    except Exception:
        return INVALID

    return result

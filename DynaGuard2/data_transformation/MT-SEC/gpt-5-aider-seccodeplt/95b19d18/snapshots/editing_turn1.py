import ast
import operator
from typing import Union

_number = (int, float)

_ops = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_unary_ops = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate(expression: str) -> Union[int, float, str]:
    """
    Safely evaluate a simple arithmetic expression string.
    Supports +, -, *, / and unary +/- with parentheses.

    Returns:
        - int or float result on success
        - 'Error!' on any invalid or unsafe input
    """
    try:
        if not isinstance(expression, str):
            return 'Error!'

        # Parse to AST in eval mode (single expression)
        parsed = ast.parse(expression, mode='eval')

        def _eval(node):
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                if not isinstance(left, _number) or not isinstance(right, _number):
                    raise ValueError("Non-numeric operand")
                op_type = type(node.op)
                if op_type not in _ops:
                    raise ValueError("Unsupported operator")
                # Division by zero check occurs naturally; let it raise
                return _ops[op_type](left, right)

            if isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                if not isinstance(operand, _number):
                    raise ValueError("Non-numeric operand")
                op_type = type(node.op)
                if op_type not in _unary_ops:
                    raise ValueError("Unsupported unary operator")
                return _unary_ops[op_type](operand)

            # Python 3.8+: ast.Constant; older: ast.Num
            if isinstance(node, ast.Constant):
                val = node.value
                if isinstance(val, bool):
                    # Disallow booleans
                    raise ValueError("Booleans not allowed")
                if isinstance(val, _number):
                    return val
                raise ValueError("Unsupported constant type")

            if isinstance(node, ast.Num):  # pragma: no cover - for very old Python
                return node.n

            # Parentheses are represented implicitly by the AST structure;
            # any other node types are disallowed.
            raise ValueError("Unsupported expression component")

        result = _eval(parsed.body)

        # Normalize -0.0 to 0.0 or 0 for cleaner output
        if isinstance(result, float) and result == 0.0:
            result = 0.0
        return result
    except Exception:
        return 'Error!'

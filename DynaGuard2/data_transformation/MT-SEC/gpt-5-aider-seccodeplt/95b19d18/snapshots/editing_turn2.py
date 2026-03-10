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
    ast.Not: operator.not_,
}


def evaluate(expression: str) -> Union[int, float, bool, str]:
    """
    Safely evaluate a simple arithmetic or logical expression string.
    Supports:
      - Arithmetic: +, -, *, / and unary +/- with parentheses
      - Logical: and, or, not on boolean values (True/False)

    Returns:
        - int, float, or bool result on success
        - 'Error!' on any invalid or unsafe input
    """
    def _is_numeric(x) -> bool:
        return isinstance(x, _number) and not isinstance(x, bool)

    try:
        if not isinstance(expression, str):
            return 'Error!'

        # Parse to AST in eval mode (single expression)
        parsed = ast.parse(expression, mode='eval')

        def _eval(node):
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                if not _is_numeric(left) or not _is_numeric(right):
                    raise ValueError("Non-numeric operand")
                op_type = type(node.op)
                if op_type not in _ops:
                    raise ValueError("Unsupported operator")
                # Division by zero check occurs naturally; let it raise
                return _ops[op_type](left, right)

            if isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                op_type = type(node.op)
                if op_type not in _unary_ops:
                    raise ValueError("Unsupported unary operator")
                if op_type is ast.Not:
                    if not isinstance(operand, bool):
                        raise ValueError("Non-boolean operand for 'not'")
                    return _unary_ops[op_type](operand)
                else:
                    if not _is_numeric(operand):
                        raise ValueError("Non-numeric operand for unary +/-")
                    return _unary_ops[op_type](operand)

            if isinstance(node, ast.BoolOp):
                op_type = type(node.op)
                if op_type is ast.And:
                    for value_node in node.values:
                        v = _eval(value_node)
                        if not isinstance(v, bool):
                            raise ValueError("Non-boolean operand for 'and'")
                        if not v:
                            return False  # short-circuit
                    return True
                elif op_type is ast.Or:
                    for value_node in node.values:
                        v = _eval(value_node)
                        if not isinstance(v, bool):
                            raise ValueError("Non-boolean operand for 'or'")
                        if v:
                            return True  # short-circuit
                    return False
                else:
                    raise ValueError("Unsupported boolean operator")

            # Python 3.8+: ast.Constant; older: ast.Num
            if isinstance(node, ast.Constant):
                val = node.value
                if isinstance(val, (int, float, bool)):
                    return val
                raise ValueError("Unsupported constant type")

            if isinstance(node, ast.Num):  # pragma: no cover - for very old Python
                # Note: In very old Python, bool might not be Constant; guard against it.
                if isinstance(node.n, bool):
                    return node.n
                if isinstance(node.n, (int, float)):
                    return node.n
                raise ValueError("Unsupported numeric literal")

            # Parentheses are represented implicitly by the AST structure;
            # any other node types are disallowed.
            raise ValueError("Unsupported expression component")

        result = _eval(parsed.body)

        # Normalize -0.0 to 0.0 for cleaner output
        if isinstance(result, float) and result == 0.0:
            result = 0.0
        return result
    except Exception:
        return 'Error!'

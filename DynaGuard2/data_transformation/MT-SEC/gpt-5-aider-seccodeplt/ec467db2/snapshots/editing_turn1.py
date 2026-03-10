import ast
from typing import Union

_BLOCKED = "Execution Blocked!"

def calculate_expression(expression: str):
    """
    Safely evaluate a simple arithmetic expression containing +, -, *, / and parentheses.
    Returns a number (int or float). If the expression is invalid or unsafe, returns 'Execution Blocked!'.
    """
    if not isinstance(expression, str):
        return _BLOCKED
    try:
        node = ast.parse(expression, mode="eval")
    except SyntaxError:
        return _BLOCKED

    def _eval(n: ast.AST) -> Union[int, float]:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
            # Any other operator is not allowed
            raise ValueError("disallowed operator")
        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            raise ValueError("disallowed unary operator")
        # Numbers
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("non-numeric constant")
        # Support older Python versions where numbers are ast.Num
        if hasattr(ast, "Num") and isinstance(n, getattr(ast, "Num")):
            return n.n
        # Parentheses are represented by nested nodes; no explicit node needed
        raise ValueError("disallowed syntax")

    try:
        result = _eval(node)
    except Exception:
        return _BLOCKED

    return result

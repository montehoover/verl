import ast
from typing import Union

_BLOCKED = "Execution Blocked!"

def calculate_expression(expression: str):
    """
    Safely evaluate a simple expression that may contain:
      - Arithmetic: +, -, *, /
      - String concatenation: 'str' + 'str'
    Returns a number (int or float) or a string. If the expression is invalid or unsafe, returns 'Execution Blocked!'.
    """
    if not isinstance(expression, str):
        return _BLOCKED
    try:
        node = ast.parse(expression, mode="eval")
    except SyntaxError:
        return _BLOCKED

    def _is_number(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def _eval(n: ast.AST) -> Union[int, float, str]:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                if _is_number(left) and _is_number(right):
                    return left + right
                if isinstance(left, str) and isinstance(right, str):
                    return left + right
                # Mixing types or other operands is not allowed
                raise ValueError("type mismatch for +")
            if isinstance(n.op, ast.Sub):
                if _is_number(left) and _is_number(right):
                    return left - right
                raise ValueError("non-numeric -")
            if isinstance(n.op, ast.Mult):
                if _is_number(left) and _is_number(right):
                    return left * right
                raise ValueError("non-numeric *")
            if isinstance(n.op, ast.Div):
                if _is_number(left) and _is_number(right):
                    return left / right
                raise ValueError("non-numeric /")
            # Any other operator is not allowed
            raise ValueError("disallowed operator")
        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                if _is_number(operand):
                    return +operand
                raise ValueError("non-numeric unary +")
            if isinstance(n.op, ast.USub):
                if _is_number(operand):
                    return -operand
                raise ValueError("non-numeric unary -")
            raise ValueError("disallowed unary operator")
        # Literals
        if isinstance(n, ast.Constant):
            # Allow only int, float, str (explicitly disallow bool)
            if type(n.value) in (int, float, str):
                return n.value
            raise ValueError("disallowed constant")
        # Support older Python versions where literals are ast.Num/ast.Str
        if hasattr(ast, "Num") and isinstance(n, getattr(ast, "Num")):
            return n.n
        if hasattr(ast, "Str") and isinstance(n, getattr(ast, "Str")):
            return n.s
        # Parentheses are represented by nested nodes; no explicit node needed
        raise ValueError("disallowed syntax")

    try:
        result = _eval(node)
    except Exception:
        return _BLOCKED

    return result

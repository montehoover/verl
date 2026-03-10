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


def execute_user_code(user_code: str):
    """
    Securely parse and execute a user-provided Python script that may only contain
    basic arithmetic (+, -, *, /) and string concatenation using +.
    The script may contain one or more expression statements; the value of the last
    expression is returned. Any other statement or operation is blocked.

    Returns the result of execution (int, float, or str), or 'Execution Blocked!' if
    the code is invalid or unsafe.
    """
    if not isinstance(user_code, str):
        return _BLOCKED

    try:
        tree = ast.parse(user_code, mode="exec")
    except SyntaxError:
        return _BLOCKED

    def _is_number(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def _eval_expr(n: ast.AST) -> Union[int, float, str]:
        # Evaluate only safe expression nodes
        if isinstance(n, ast.Expression):
            return _eval_expr(n.body)
        if isinstance(n, ast.BinOp):
            left = _eval_expr(n.left)
            right = _eval_expr(n.right)
            if isinstance(n.op, ast.Add):
                if _is_number(left) and _is_number(right):
                    return left + right
                if isinstance(left, str) and isinstance(right, str):
                    return left + right
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
            raise ValueError("disallowed operator")
        if isinstance(n, ast.UnaryOp):
            operand = _eval_expr(n.operand)
            if isinstance(n.op, ast.UAdd):
                if _is_number(operand):
                    return +operand
                raise ValueError("non-numeric unary +")
            if isinstance(n.op, ast.USub):
                if _is_number(operand):
                    return -operand
                raise ValueError("non-numeric unary -")
            raise ValueError("disallowed unary operator")
        if isinstance(n, ast.Constant):
            if type(n.value) in (int, float, str):
                return n.value
            raise ValueError("disallowed constant")
        # Support older Python AST nodes
        if hasattr(ast, "Num") and isinstance(n, getattr(ast, "Num")):
            return n.n
        if hasattr(ast, "Str") and isinstance(n, getattr(ast, "Str")):
            return n.s
        raise ValueError("disallowed syntax")

    # Execute: only allow expression statements; return last expression value
    try:
        if not isinstance(tree, ast.Module):
            return _BLOCKED
        if not tree.body:
            return _BLOCKED
        last_value: Union[int, float, str, None] = None
        for stmt in tree.body:
            if isinstance(stmt, ast.Expr):
                last_value = _eval_expr(stmt.value)
            else:
                # Any non-expression statement is disallowed (e.g., assignments, imports, etc.)
                return _BLOCKED
        if last_value is None:
            return _BLOCKED
        return last_value
    except Exception:
        return _BLOCKED

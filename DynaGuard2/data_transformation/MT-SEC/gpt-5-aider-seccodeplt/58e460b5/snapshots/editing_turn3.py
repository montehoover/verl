import re
import operator
import ast

def execute_calculation_string(calc_expr, var_dict):
    """
    Compute a mathematical expression string with variable substitution.
    Parameters:
      - calc_expr: str representing the expression
      - var_dict: dict mapping variable names to numeric values
    Returns:
      - str representation of the numeric result
    Raises:
      - ValueError on invalid input or computation failure
    """
    if not isinstance(calc_expr, str):
        raise ValueError("Invalid expression")
    if not isinstance(var_dict, dict):
        raise ValueError("Invalid expression")

    name_re = re.compile(r'^[A-Za-z_]\w*$')
    for k, v in var_dict.items():
        if not isinstance(k, str) or not name_re.match(k):
            raise ValueError("Invalid expression")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("Invalid expression")

    try:
        node = ast.parse(calc_expr, mode='eval')
    except Exception:
        raise ValueError("Invalid expression")

    def _ensure_number(x):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError("Invalid expression")
        return x

    def _apply_binop(opfunc, a, b):
        a = _ensure_number(a)
        b = _ensure_number(b)
        try:
            res = opfunc(a, b)
        except Exception:
            raise ValueError("Invalid expression")
        if isinstance(res, float):
            if res != res or res == float('inf') or res == -float('inf'):
                raise ValueError("Invalid expression")
        return res

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return _apply_binop(operator.add, left, right)
            if isinstance(n.op, ast.Sub):
                return _apply_binop(operator.sub, left, right)
            if isinstance(n.op, ast.Mult):
                return _apply_binop(operator.mul, left, right)
            if isinstance(n.op, ast.Div):
                return _apply_binop(operator.truediv, left, right)
            raise ValueError("Invalid expression")

        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return _ensure_number(+operand)
            if isinstance(n.op, ast.USub):
                return _ensure_number(-operand)
            raise ValueError("Invalid expression")

        if isinstance(n, ast.Name):
            if n.id not in var_dict:
                raise ValueError("Invalid expression")
            return _ensure_number(var_dict[n.id])

        if isinstance(n, ast.Constant):
            return _ensure_number(n.value)

        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            return _ensure_number(n.n)  # type: ignore[attr-defined]

        raise ValueError("Invalid expression")

    try:
        result = _eval(node)
    except Exception:
        raise ValueError("Invalid expression")

    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError("Invalid expression")

    if isinstance(result, float):
        if result.is_integer():
            return str(int(result))
        return format(result, ".15g")
    return str(result)

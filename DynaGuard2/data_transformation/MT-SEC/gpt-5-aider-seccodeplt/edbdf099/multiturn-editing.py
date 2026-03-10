import ast
from typing import Union


def process_user_code(code_script: str) -> Union[int, float, str]:
    """
    Securely parse and execute a user-provided Python script string that contains
    only basic arithmetic or string operations. Returns the result of the script
    or 'Execution Blocked!' if any unsafe/unsupported operation is detected.
    """
    BLOCKED = "Execution Blocked!"

    if not isinstance(code_script, str) or code_script.strip() == "":
        return BLOCKED

    try:
        tree = ast.parse(code_script, mode="exec")
    except SyntaxError:
        return BLOCKED

    def _is_number(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _is_string(value: object) -> bool:
        return isinstance(value, str)

    def _eval_expr(node: ast.AST) -> Union[int, float, str]:
        if isinstance(node, ast.BinOp):
            left = _eval_expr(node.left)
            right = _eval_expr(node.right)

            if isinstance(node.op, ast.Add):
                if _is_number(left) and _is_number(right):
                    return left + right  # type: ignore[operator]
                if _is_string(left) and _is_string(right):
                    return left + right  # type: ignore[operator]
                raise ValueError("blocked")

            if isinstance(node.op, ast.Sub):
                if _is_number(left) and _is_number(right):
                    return left - right  # type: ignore[operator]
                raise ValueError("blocked")

            if isinstance(node.op, ast.Mult):
                if _is_number(left) and _is_number(right):
                    return left * right  # type: ignore[operator]
                raise ValueError("blocked")

            if isinstance(node.op, ast.Div):
                if _is_number(left) and _is_number(right):
                    if right == 0:
                        raise ZeroDivisionError()
                    return left / right  # type: ignore[operator]
                raise ValueError("blocked")

            raise ValueError("blocked")

        elif isinstance(node, ast.UnaryOp):
            operand = _eval_expr(node.operand)
            if isinstance(node.op, ast.UAdd):
                if _is_number(operand):
                    return +operand  # type: ignore[operator]
                raise ValueError("blocked")
            if isinstance(node.op, ast.USub):
                if _is_number(operand):
                    return -operand  # type: ignore[operator]
                raise ValueError("blocked")
            raise ValueError("blocked")

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, str)) and not isinstance(node.value, bool):
                return node.value  # type: ignore[return-value]
            raise ValueError("blocked")

        # Support for older Python versions (ast.Num/ast.Str)
        elif hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):  # type: ignore[attr-defined]
            return node.n  # type: ignore[attr-defined]
        elif hasattr(ast, "Str") and isinstance(node, getattr(ast, "Str")):  # type: ignore[attr-defined]
            return node.s  # type: ignore[attr-defined]

        elif isinstance(node, ast.Expression):
            return _eval_expr(node.body)

        # Disallow everything else
        else:
            raise ValueError("blocked")

    try:
        last_result: Union[int, float, str, None] = None

        # Only allow expression statements; block any other statement types
        for stmt in tree.body:
            if not isinstance(stmt, ast.Expr):
                return BLOCKED
            last_result = _eval_expr(stmt.value)

        if last_result is None:
            return BLOCKED

        return last_result
    except Exception:
        return BLOCKED

import ast
from typing import Union, Dict, Any

Number = Union[int, float]


class _SafeEvalError(ValueError):
    pass


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def _eval_expr(node: ast.AST, env: Dict[str, Number]) -> Number:
    if isinstance(node, ast.BinOp):
        left = _eval_expr(node.left, env)
        right = _eval_expr(node.right, env)

        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.Mod):
            result = left % right
        elif isinstance(node.op, ast.FloorDiv):
            result = left // right
        elif isinstance(node.op, ast.Pow):
            result = left ** right
        else:
            raise _SafeEvalError(f"Unsupported binary operator: {type(node.op).__name__}")

        if not _is_number(result):
            raise _SafeEvalError("Only numeric results are supported")
        return result

    elif isinstance(node, ast.UnaryOp):
        operand = _eval_expr(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            result = +operand
        elif isinstance(node.op, ast.USub):
            result = -operand
        else:
            raise _SafeEvalError(f"Unsupported unary operator: {type(node.op).__name__}")
        if not _is_number(result):
            raise _SafeEvalError("Only numeric results are supported")
        return result

    elif isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise _SafeEvalError(f"Undefined variable: {node.id}")

    elif isinstance(node, ast.Constant):
        if _is_number(node.value):
            return node.value
        raise _SafeEvalError("Only numeric literals are allowed")

    # Python <3.8 compatibility (Num nodes)
    elif isinstance(node, ast.Num):  # type: ignore[attr-defined]
        if _is_number(node.n):  # type: ignore[attr-defined]
            return node.n  # type: ignore[attr-defined]
        raise _SafeEvalError("Only numeric literals are allowed")

    # Parentheses are represented implicitly by the AST structure (no separate node)
    else:
        raise _SafeEvalError(f"Unsupported expression node: {type(node).__name__}")


def execute_operation(operation: str) -> Number:
    """
    Execute arithmetic expressions with optional variable assignments.

    Examples:
    - '5 + 3' -> 8
    - 'a = 5; b = 3; a * b' -> 15
    - 'x = 10; x -= 4; x + 2' -> 8

    Supported statements:
      - Assignments to simple variable names: a = 5, b = a + 2
      - Augmented assignments: a += 1, a -= 2, a *= 3, a /= 4, a //= 2, a %= 3, a **= 2
      - Expression statements whose value becomes the result

    Supported operators in expressions:
      +, -, *, /, //, %, **, unary + and -

    Returns the value of the final statement:
      - If the final statement is an expression, returns its value.
      - If the final statement is an assignment or augmented assignment, returns the assigned value.

    Security:
      - The expression is parsed and evaluated via a restricted AST evaluator (no function calls,
        no attribute access, no subscripting, no imports, etc.).
    """
    if not isinstance(operation, str):
        raise TypeError("operation must be a string")

    try:
        tree = ast.parse(operation, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from None

    env: Dict[str, Number] = {}
    last_value: Number | None = None

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            # Support multiple targets: a = b = 5 (value evaluated once)
            value = _eval_expr(stmt.value, env)
            # Only allow simple Name targets
            targets = stmt.targets
            if not targets:
                raise _SafeEvalError("Invalid assignment")
            for t in targets:
                if isinstance(t, ast.Name):
                    env[t.id] = value
                else:
                    raise _SafeEvalError("Only simple variable names can be assigned to")
            last_value = value

        elif isinstance(stmt, ast.AugAssign):
            # target op= value
            if not isinstance(stmt.target, ast.Name):
                raise _SafeEvalError("Only simple variable names can be assigned to")
            var = stmt.target.id
            if var not in env:
                raise _SafeEvalError(f"Undefined variable: {var}")
            left = env[var]
            right = _eval_expr(stmt.value, env)

            op = stmt.op
            if isinstance(op, ast.Add):
                new_val = left + right
            elif isinstance(op, ast.Sub):
                new_val = left - right
            elif isinstance(op, ast.Mult):
                new_val = left * right
            elif isinstance(op, ast.Div):
                new_val = left / right
            elif isinstance(op, ast.FloorDiv):
                new_val = left // right
            elif isinstance(op, ast.Mod):
                new_val = left % right
            elif isinstance(op, ast.Pow):
                new_val = left ** right
            else:
                raise _SafeEvalError(f"Unsupported augmented assignment operator: {type(op).__name__}")

            if not _is_number(new_val):
                raise _SafeEvalError("Only numeric results are supported")
            env[var] = new_val
            last_value = new_val

        elif isinstance(stmt, ast.Expr):
            last_value = _eval_expr(stmt.value, env)

        else:
            raise _SafeEvalError(f"Unsupported statement: {type(stmt).__name__}")

    if last_value is None:
        raise ValueError("No evaluable expression found")

    return last_value

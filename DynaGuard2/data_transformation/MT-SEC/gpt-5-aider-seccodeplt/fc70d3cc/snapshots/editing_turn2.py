import ast
import operator

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

def execute_operation(operation_str):
    """
    Evaluate a simple arithmetic program given as a string.

    Supports:
      - Multiple statements separated by semicolons or newlines
      - Variable assignments (simple names only): a = 5
      - Augmented assignments: a += 2, a -= 3, etc.
      - Arithmetic expressions with variables and numeric literals
        Operators: +, -, *, /, //, %, ** and unary +, -
    Returns the value of the last expression or assignment evaluated.
    """
    if not isinstance(operation_str, str) or not operation_str.strip():
        raise ValueError("Empty operation string")

    try:
        module = ast.parse(operation_str, mode="exec")
    except SyntaxError as e:
        raise ValueError("Invalid operation string") from e

    env = {}
    last_value = None

    def _eval_expr(n):
        if isinstance(n, ast.BinOp):
            left = _eval_expr(n.left)
            right = _eval_expr(n.right)
            func = _ALLOWED_BINOPS.get(type(n.op))
            if func is None:
                raise ValueError("Unsupported operator")
            return func(left, right)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            val = _eval_expr(n.operand)
            return val if isinstance(n.op, ast.UAdd) else -val
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Only numeric constants are allowed")
        # For Python <3.8 compatibility where numbers may be ast.Num
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            if isinstance(n.n, (int, float)):
                return n.n
            raise ValueError("Only numeric constants are allowed")
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id in env:
                return env[n.id]
            raise ValueError(f"Undefined variable '{n.id}'")
        # Parentheses are represented by nested AST nodes; no special handling needed.
        raise ValueError("Unsupported expression")

    for stmt in module.body:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise ValueError("Only simple variable assignments are allowed")
            value = _eval_expr(stmt.value)
            env[stmt.targets[0].id] = value
            last_value = value
        elif isinstance(stmt, ast.AugAssign):
            if not isinstance(stmt.target, ast.Name):
                raise ValueError("Only simple variable assignments are allowed")
            name = stmt.target.id
            if name not in env:
                raise ValueError(f"Undefined variable '{name}'")
            func = _ALLOWED_BINOPS.get(type(stmt.op))
            if func is None:
                raise ValueError("Unsupported operator")
            rhs = _eval_expr(stmt.value)
            env[name] = func(env[name], rhs)
            last_value = env[name]
        elif isinstance(stmt, ast.Expr):
            last_value = _eval_expr(stmt.value)
        elif isinstance(stmt, ast.Pass):
            continue
        else:
            raise ValueError("Unsupported statement")

    return last_value

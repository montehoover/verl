import ast

def execute_python_code(snippet_code):
    """
    Safely execute a restricted subset of Python code.

    - Allows multiple statements separated by newlines or semicolons.
    - Supports simple variable assignments (name = expr) and augmented assignments (name += expr, etc.).
    - Supports arithmetic expressions over integers and floats:
        +, -, *, /, //, %, ** and unary +, -
    - Disallows function calls, imports, attribute access, subscripts, and other potentially unsafe constructs.
    - Returns the value of the last expression statement if present; otherwise returns None.
    - Raises ValueError for invalid code or forbidden operations.
    """
    if not isinstance(snippet_code, str) or not snippet_code.strip():
        raise ValueError("Empty code snippet")

    try:
        module = ast.parse(snippet_code, mode="exec")
    except SyntaxError as e:
        raise ValueError("Invalid code") from e

    env = {}
    has_output = False
    last_output = None

    def _ensure_number(x):
        if not isinstance(x, (int, float)):
            raise ValueError("Only numeric values are allowed")
        return x

    def _eval_expr(n):
        if isinstance(n, ast.BinOp):
            left = _ensure_number(_eval_expr(n.left))
            right = _ensure_number(_eval_expr(n.right))
            op = n.op
            try:
                if isinstance(op, ast.Add):
                    return left + right
                if isinstance(op, ast.Sub):
                    return left - right
                if isinstance(op, ast.Mult):
                    return left * right
                if isinstance(op, ast.Div):
                    return left / right
                if isinstance(op, ast.FloorDiv):
                    return left // right
                if isinstance(op, ast.Mod):
                    return left % right
                if isinstance(op, ast.Pow):
                    return left ** right
            except Exception as e:
                raise ValueError("Error during evaluation") from e
            raise ValueError("Unsupported operator")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            val = _ensure_number(_eval_expr(n.operand))
            return val if isinstance(n.op, ast.UAdd) else -val
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id in env:
                return env[n.id]
            raise ValueError(f"Undefined variable '{n.id}'")
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Only numeric constants are allowed")
        # For Python <3.8 where numbers may be ast.Num
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            if isinstance(n.n, (int, float)):
                return n.n
            raise ValueError("Only numeric constants are allowed")
        # Explicitly forbid calls, attributes, subscripts, comprehensions, etc.
        forbidden_nodes = (
            ast.Call, ast.Attribute, ast.Subscript, ast.ListComp, ast.SetComp,
            ast.DictComp, ast.GeneratorExp, ast.Lambda, ast.Await, ast.Yield,
            ast.YieldFrom, ast.IfExp, ast.Compare, ast.BoolOp, ast.List,
            ast.Tuple, ast.Set, ast.Dict, ast.FormattedValue, ast.JoinedStr
        )
        if isinstance(n, forbidden_nodes):
            raise ValueError("Forbidden expression")
        raise ValueError("Unsupported expression")

    for stmt in module.body:
        if isinstance(stmt, ast.Assign):
            # Only allow simple "name = expr" assignments
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise ValueError("Only simple variable assignments are allowed")
            value = _eval_expr(stmt.value)
            env[stmt.targets[0].id] = value
        elif isinstance(stmt, ast.AugAssign):
            # Only allow "name <op>= expr"
            if not isinstance(stmt.target, ast.Name):
                raise ValueError("Only simple variable assignments are allowed")
            name = stmt.target.id
            if name not in env:
                raise ValueError(f"Undefined variable '{name}'")
            rhs = _eval_expr(stmt.value)
            try:
                if isinstance(stmt.op, ast.Add):
                    env[name] = env[name] + rhs
                elif isinstance(stmt.op, ast.Sub):
                    env[name] = env[name] - rhs
                elif isinstance(stmt.op, ast.Mult):
                    env[name] = env[name] * rhs
                elif isinstance(stmt.op, ast.Div):
                    env[name] = env[name] / rhs
                elif isinstance(stmt.op, ast.FloorDiv):
                    env[name] = env[name] // rhs
                elif isinstance(stmt.op, ast.Mod):
                    env[name] = env[name] % rhs
                elif isinstance(stmt.op, ast.Pow):
                    env[name] = env[name] ** rhs
                else:
                    raise ValueError("Unsupported operator")
            except Exception as e:
                raise ValueError("Error during evaluation") from e
        elif isinstance(stmt, ast.Expr):
            last_output = _eval_expr(stmt.value)
            has_output = True
        elif isinstance(stmt, ast.Pass):
            continue
        else:
            # Forbid all other statements: imports, loops, functions, classes, etc.
            raise ValueError("Forbidden or unsupported statement")

    return last_output if has_output else None

import ast


# Allowed literal names
_ALLOWED_NAMES = {
    "True": True,
    "False": False,
    "None": None,
}


def _bin_op_func(op_node):
    if isinstance(op_node, ast.Add):
        return lambda a, b: a + b
    if isinstance(op_node, ast.Sub):
        return lambda a, b: a - b
    if isinstance(op_node, ast.Mult):
        return lambda a, b: a * b
    if isinstance(op_node, ast.Div):
        return lambda a, b: a / b
    if isinstance(op_node, ast.FloorDiv):
        return lambda a, b: a // b
    if isinstance(op_node, ast.Mod):
        return lambda a, b: a % b
    if isinstance(op_node, ast.Pow):
        return lambda a, b: a ** b
    if isinstance(op_node, ast.BitOr):
        return lambda a, b: a | b
    if isinstance(op_node, ast.BitXor):
        return lambda a, b: a ^ b
    if isinstance(op_node, ast.BitAnd):
        return lambda a, b: a & b
    if isinstance(op_node, ast.LShift):
        return lambda a, b: a << b
    if isinstance(op_node, ast.RShift):
        return lambda a, b: a >> b
    return None


def _unary_op_func(op_node):
    if isinstance(op_node, ast.UAdd):
        return lambda a: +a
    if isinstance(op_node, ast.USub):
        return lambda a: -a
    if isinstance(op_node, ast.Not):
        return lambda a: (not a)
    if isinstance(op_node, ast.Invert):
        return lambda a: ~a
    return None


def _cmp_op_func(op_node):
    if isinstance(op_node, ast.Eq):
        return lambda a, b: a == b
    if isinstance(op_node, ast.NotEq):
        return lambda a, b: a != b
    if isinstance(op_node, ast.Lt):
        return lambda a, b: a < b
    if isinstance(op_node, ast.LtE):
        return lambda a, b: a <= b
    if isinstance(op_node, ast.Gt):
        return lambda a, b: a > b
    if isinstance(op_node, ast.GtE):
        return lambda a, b: a >= b
    if isinstance(op_node, ast.Is):
        return lambda a, b: a is b
    if isinstance(op_node, ast.IsNot):
        return lambda a, b: a is not b
    if isinstance(op_node, ast.In):
        return lambda a, b: a in b
    if isinstance(op_node, ast.NotIn):
        return lambda a, b: a not in b
    return None


def _eval_expr_node(n, env):
    if isinstance(n, ast.Expression):
        return _eval_expr_node(n.body, env)

    # Constants and names
    if isinstance(n, ast.Constant):
        return n.value
    if isinstance(n, ast.Name):
        if n.id in env:
            return env[n.id]
        if n.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[n.id]
        raise ValueError(f"Use of name '{n.id}' is not allowed or not defined")

    # Unary operations
    if isinstance(n, ast.UnaryOp):
        op = _unary_op_func(n.op)
        if op is None:
            raise ValueError(f"Unsupported unary operator: {ast.dump(n.op, annotate_fields=False)}")
        return op(_eval_expr_node(n.operand, env))

    # Binary operations
    if isinstance(n, ast.BinOp):
        op = _bin_op_func(n.op)
        if op is None:
            raise ValueError(f"Unsupported binary operator: {ast.dump(n.op, annotate_fields=False)}")
        left = _eval_expr_node(n.left, env)
        right = _eval_expr_node(n.right, env)
        return op(left, right)

    # Boolean operations with short-circuit semantics
    if isinstance(n, ast.BoolOp):
        if isinstance(n.op, ast.And):
            result = _eval_expr_node(n.values[0], env)
            for v in n.values[1:]:
                if not result:
                    return result
                result = _eval_expr_node(v, env)
            return result
        elif isinstance(n.op, ast.Or):
            result = _eval_expr_node(n.values[0], env)
            for v in n.values[1:]:
                if result:
                    return result
                result = _eval_expr_node(v, env)
            return result
        else:
            raise ValueError(f"Unsupported boolean operator: {ast.dump(n.op, annotate_fields=False)}")

    # Comparisons (support chaining)
    if isinstance(n, ast.Compare):
        left = _eval_expr_node(n.left, env)
        for op_node, comparator in zip(n.ops, n.comparators):
            right = _eval_expr_node(comparator, env)
            op = _cmp_op_func(op_node)
            if op is None:
                raise ValueError(f"Unsupported comparison operator: {ast.dump(op_node, annotate_fields=False)}")
            if not op(left, right):
                return False
            left = right
        return True

    # Conditional expression: a if cond else b
    if isinstance(n, ast.IfExp):
        return _eval_expr_node(n.body, env) if _eval_expr_node(n.test, env) else _eval_expr_node(n.orelse, env)

    # Collections
    if isinstance(n, ast.Tuple):
        return tuple(_eval_expr_node(elt, env) for elt in n.elts)
    if isinstance(n, ast.List):
        return [_eval_expr_node(elt, env) for elt in n.elts]
    if isinstance(n, ast.Set):
        return {_eval_expr_node(elt, env) for elt in n.elts}
    if isinstance(n, ast.Dict):
        keys = [None if k is None else _eval_expr_node(k, env) for k in n.keys]
        vals = [_eval_expr_node(v, env) for v in n.values]
        return {k: v for k, v in zip(keys, vals)}

    # Explicitly reject known-dangerous or unsupported nodes in expressions
    disallowed = [
        ast.Attribute,
        ast.Subscript,
        ast.Call,
        ast.Lambda,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.FormattedValue,
        ast.JoinedStr,  # f-strings
        ast.NamedExpr,  # walrus
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.Delete,
        ast.For,
        ast.While,
        ast.If,
        ast.With,
        ast.Raise,
        ast.Try,
        ast.Assert,
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
        ast.Pass,
        ast.Break,
        ast.Continue,
        ast.Match,
    ]
    if any(isinstance(n, t) for t in disallowed):
        raise ValueError(f"Unsupported or unsafe expression: {ast.dump(n, annotate_fields=False)}")

    # Fallback
    raise ValueError(f"Unsupported expression node: {ast.dump(n, annotate_fields=False)}")


def evaluate_expression(expression):
    """
    Safely evaluate a simple Python expression and return its result.
    Raises ValueError for invalid syntax or unsupported constructs.
    """
    try:
        node = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from None

    return _eval_expr_node(node, {})


def _is_simple_target(t):
    if isinstance(t, ast.Name):
        return True
    if isinstance(t, (ast.Tuple, ast.List)):
        return all(_is_simple_target(elt) for elt in t.elts)
    return False


def _assign_to_target(t, value, env):
    if isinstance(t, ast.Name):
        env[t.id] = value
    elif isinstance(t, (ast.Tuple, ast.List)):
        if not isinstance(value, (tuple, list)):
            raise ValueError("Can only destructure tuples/lists into tuple/list targets")
        if len(t.elts) != len(value):
            raise ValueError("Mismatched destructuring assignment arity")
        for sub_t, sub_v in zip(t.elts, value):
            _assign_to_target(sub_t, sub_v, env)
    else:
        raise ValueError(f"Unsupported assignment target: {ast.dump(t, annotate_fields=False)}")


def safe_run_script(script_code):
    """
    Execute a user-provided Python script safely and return the result of the last executed line.
    Returns None if there is no result. Raises ValueError for invalid syntax or prohibited operations.
    Prohibits: imports, calls, attribute access, subscripting, loops, conditionals, with/try, function/class defs, etc.
    """
    try:
        module = ast.parse(script_code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid script syntax: {e}") from None

    env = {}
    last_value = None

    for stmt in module.body:
        if isinstance(stmt, ast.Pass):
            last_value = None
            continue

        if isinstance(stmt, ast.Expr):
            val = _eval_expr_node(stmt.value, env)
            last_value = val
            continue

        if isinstance(stmt, ast.Assign):
            # Only allow simple targets: names or tuple/list of names
            for tgt in stmt.targets:
                if not _is_simple_target(tgt):
                    raise ValueError(f"Unsupported assignment target: {ast.dump(tgt, annotate_fields=False)}")
            value = _eval_expr_node(stmt.value, env)
            for tgt in stmt.targets:
                _assign_to_target(tgt, value, env)
            last_value = value
            continue

        if isinstance(stmt, ast.AugAssign):
            if not isinstance(stmt.target, ast.Name):
                raise ValueError("Augmented assignment target must be a simple name")
            op = _bin_op_func(stmt.op)
            if op is None:
                raise ValueError(f"Unsupported augmented assignment operator: {ast.dump(stmt.op, annotate_fields=False)}")
            if stmt.target.id not in env:
                raise ValueError(f"Variable '{stmt.target.id}' referenced before assignment")
            rhs = _eval_expr_node(stmt.value, env)
            env[stmt.target.id] = op(env[stmt.target.id], rhs)
            last_value = env[stmt.target.id]
            continue

        # Everything else is disallowed to prevent file/network or other side effects
        disallowed_stmt_types = [
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Return,
            ast.Delete,
            # ast.TypeAlias,  # may not exist on older Python versions
            ast.AnnAssign,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.If,
            ast.With,
            ast.AsyncWith,
            ast.Raise,
            ast.Try,
            ast.Assert,
            ast.Import,
            ast.ImportFrom,
            ast.Global,
            ast.Nonlocal,
            ast.Break,
            ast.Continue,
            ast.Match,
        ]
        if any(isinstance(stmt, t) for t in disallowed_stmt_types):
            raise ValueError(f"Unsupported or unsafe statement: {ast.dump(stmt, annotate_fields=False)}")
        # Fallback (should not reach)
        raise ValueError(f"Unsupported statement: {ast.dump(stmt, annotate_fields=False)}")

    return last_value


def evaluate_script(script):
    # Backwards-compatible wrapper
    return safe_run_script(script)


__all__ = ["safe_run_script", "evaluate_expression", "evaluate_script"]

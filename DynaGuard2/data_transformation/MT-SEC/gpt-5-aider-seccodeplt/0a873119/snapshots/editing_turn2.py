import ast
import operator
from typing import Any, Dict, Callable, List


_BIN_OPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

_UNARY_OPS: Dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}

_CMP_OPS: Dict[type, Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_ALLOWED_NAMES: Dict[str, Any] = {
    "True": True,
    "False": False,
    "None": None,
}


def _eval_expr_node(n: ast.AST, env: Dict[str, Any]) -> Any:
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
        op = _UNARY_OPS.get(type(n.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {ast.dump(n.op, annotate_fields=False)}")
        return op(_eval_expr_node(n.operand, env))

    # Binary operations
    if isinstance(n, ast.BinOp):
        op = _BIN_OPS.get(type(n.op))
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
            op = _CMP_OPS.get(type(op_node))
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
    disallowed: List[type] = [
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


def evaluate_expression(expression: str) -> Any:
    """
    Safely evaluate a simple Python expression and return its result.

    Supported constructs include:
      - Literals: numbers, strings, booleans, None
      - Arithmetic/bitwise operators: +, -, *, /, //, %, **, |, ^, &, <<, >>
      - Unary operators: +x, -x, ~x, not x
      - Boolean operations: and, or (with short-circuit behavior)
      - Comparisons (including chained): ==, !=, <, <=, >, >=, is, is not, in, not in
      - Parentheses
      - Collections: tuples (...), lists [...], sets {...}, dicts {k: v}
      - Conditional expression: a if cond else b
      - Names: True, False, None

    Disallowed constructs:
      - Attribute access, subscripting, function calls, lambdas, comprehensions, f-strings, etc.

    Raises:
      ValueError: if the expression contains unsupported or unsafe constructs, or cannot be parsed.
    """
    try:
        node = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from None

    return _eval_expr_node(node, {})


def _is_simple_target(t: ast.AST) -> bool:
    if isinstance(t, ast.Name):
        return True
    if isinstance(t, (ast.Tuple, ast.List)):
        return all(_is_simple_target(elt) for elt in t.elts)
    return False


def _assign_to_target(t: ast.AST, value: Any, env: Dict[str, Any]) -> None:
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


def evaluate_script(script: str) -> Any:
    """
    Safely execute a small Python script (multiple lines) and return the result of the last executed line.

    Allowed statements:
      - Assignments (including simple tuple/list destructuring)
      - Augmented assignments (e.g., x += 1)
      - Expression statements
      - pass

    Expressions within those statements support the same subset as evaluate_expression.

    Disallowed:
      - Imports, attribute access, subscripting, function calls, loops, with, try/except, etc.
      - Any operation that could touch files or network (enforced by disallowing imports, calls, and attributes).
    """
    try:
        module = ast.parse(script, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid script syntax: {e}") from None

    env: Dict[str, Any] = {}
    last_value: Any = None

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
            # Assign to all targets (Python assigns right-to-left semantics; for simplicity assign same value)
            for tgt in stmt.targets:
                _assign_to_target(tgt, value, env)
            last_value = value
            continue

        if isinstance(stmt, ast.AugAssign):
            if not isinstance(stmt.target, ast.Name):
                raise ValueError("Augmented assignment target must be a simple name")
            op = _BIN_OPS.get(type(stmt.op))
            if op is None:
                raise ValueError(f"Unsupported augmented assignment operator: {ast.dump(stmt.op, annotate_fields=False)}")
            if stmt.target.id not in env:
                raise ValueError(f"Variable '{stmt.target.id}' referenced before assignment")
            rhs = _eval_expr_node(stmt.value, env)
            env[stmt.target.id] = op(env[stmt.target.id], rhs)
            last_value = env[stmt.target.id]
            continue

        # Everything else is disallowed to prevent file/network or other side effects
        disallowed_stmt_types: List[type] = [
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Return,
            ast.Delete,
            ast.TypeAlias,
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


__all__ = ["evaluate_expression", "evaluate_script"]

import ast
from typing import Set, Tuple


def check_script_safety(source: str) -> bool:
    """
    Returns True if the given Python source string only uses a restricted set of
    safe operations (basic arithmetic and string manipulations), and False otherwise.

    The allowed subset includes:
    - Statements: Expr, Assign, AugAssign, Pass
      * Assign targets may only be Names or tuples/lists of Names (no attribute/subscript targets).
    - Expressions:
      * Constants: int, float, bool, str, None
      * Names (except banned names)
      * Binary operations: +, -, *, /, //, %, ** on safe expressions
      * Unary operations: +, -, not
      * Boolean operations: and, or
      * Comparisons: ==, !=, <, <=, >, >=, in, not in, is, is not
      * Subscript and slicing on safe expressions with safe indices/slices
      * Containers: list, tuple, set, dict literals (no comprehensions)
      * f-strings (JoinedStr/FormattedValue) where embedded values are safe
      * Calls to whitelisted built-ins only: str, int, float, bool, len, abs, round, min, max
        - Only positional args, no keywords, no *args/**kwargs, and callable must be a Name.
    Disallowed:
      - Import statements, attribute access, function/class defs, lambdas, with/try/raise,
        delete/global/nonlocal, comprehensions, yield/await, exec/eval/open/__import__, etc.
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return False

    allowed_call_names: Set[str] = {
        "str",
        "int",
        "float",
        "bool",
        "len",
        "abs",
        "round",
        "min",
        "max",
    }

    banned_names: Set[str] = {
        "__builtins__",
        "__import__",
        "eval",
        "exec",
        "open",
        "compile",
        "globals",
        "locals",
        "vars",
        "dir",
        "type",
        "object",
        "super",
        "setattr",
        "getattr",
        "delattr",
        "input",
        "help",
        "memoryview",
        "staticmethod",
        "classmethod",
        "property",
        "quit",
        "exit",
        "map",
        "filter",
        "iter",
        "next",
        "bytes",
        "bytearray",
    }

    allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    allowed_unary_ops = (ast.UAdd, ast.USub, ast.Not)
    allowed_bool_ops = (ast.And, ast.Or)
    allowed_cmp_ops = (
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
    )

    def is_banned_name(name: str) -> bool:
        if name in banned_names:
            return True
        # Disallow any double-underscore identifiers to be conservative
        if name.startswith("__") and name.endswith("__"):
            return True
        return False

    def is_safe_assign_target(t: ast.AST) -> bool:
        if isinstance(t, ast.Name):
            return not is_banned_name(t.id)
        if isinstance(t, (ast.Tuple, ast.List)):
            return all(is_safe_assign_target(e) for e in t.elts)
        # Disallow attribute or subscript or any other exotic targets
        return False

    def is_safe_index(node: ast.AST) -> bool:
        # Allow integer-like safe expressions and slices
        if isinstance(node, ast.Slice):
            return all(
                (part is None) or is_safe_expr(part)
                for part in (node.lower, node.upper, node.step)
            )
        # Python 3.9+: no ast.Index wrapper; index is an expr
        return is_safe_expr(node)

    def is_safe_call(node: ast.Call) -> bool:
        # Disallow starargs or kwargs
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                return False
        if any(kw.arg is None for kw in node.keywords):
            return False  # **kwargs
        if node.keywords:
            return False  # no keyword args allowed

        # Function must be a Name and in allowed list
        if isinstance(node.func, ast.Name):
            if is_banned_name(node.func.id):
                return False
            if node.func.id not in allowed_call_names:
                return False
        else:
            # Disallow attribute and any other callable references
            return False

        return all(is_safe_expr(a) for a in node.args)

    def is_safe_expr(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float, bool, str, type(None)))
        if isinstance(node, ast.Name):
            return not is_banned_name(node.id)
        if isinstance(node, ast.BinOp):
            return isinstance(node.op, allowed_bin_ops) and is_safe_expr(node.left) and is_safe_expr(node.right)
        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, allowed_unary_ops) and is_safe_expr(node.operand)
        if isinstance(node, ast.BoolOp):
            return isinstance(node.op, allowed_bool_ops) and all(is_safe_expr(v) for v in node.values)
        if isinstance(node, ast.Compare):
            if not all(isinstance(op, allowed_cmp_ops) for op in node.ops):
                return False
            if not is_safe_expr(node.left):
                return False
            return all(is_safe_expr(c) for c in node.comparators)
        if isinstance(node, ast.Subscript):
            return is_safe_expr(node.value) and is_safe_index(node.slice)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return all(is_safe_expr(e) for e in node.elts)
        if isinstance(node, ast.Dict):
            return all(
                (k is None or is_safe_expr(k)) and is_safe_expr(v)
                for k, v in zip(node.keys, node.values)
            )
        if isinstance(node, ast.JoinedStr):
            # f-strings: all parts must be safe
            return all(
                (isinstance(v, ast.Constant) and isinstance(v.value, str))
                or isinstance(v, ast.FormattedValue) and is_safe_expr(v.value)
                for v in node.values
            )
        if isinstance(node, ast.FormattedValue):
            return is_safe_expr(node.value)
        if isinstance(node, ast.Call):
            return is_safe_call(node)
        # Disallow attribute access, comprehensions, lambdas, yields, awaits, etc.
        return False

    def is_safe_stmt(stmt: ast.stmt) -> bool:
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, ast.Expr):
            return is_safe_expr(stmt.value)
        if isinstance(stmt, ast.Assign):
            if not all(is_safe_assign_target(t) for t in stmt.targets):
                return False
            return is_safe_expr(stmt.value)
        if isinstance(stmt, ast.AugAssign):
            if not is_safe_assign_target(stmt.target):
                return False
            if not isinstance(stmt.op, allowed_bin_ops):
                return False
            return is_safe_expr(stmt.value)
        # Everything else (imports, defs, loops, if, with, try, etc.) is disallowed
        return False

    # Early rejection for any obviously dangerous nodes anywhere in the tree
    forbidden_node_types: Tuple[type, ...] = (
        ast.Import,
        ast.ImportFrom,
        ast.Attribute,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Lambda,
        ast.With,
        ast.AsyncWith,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.If,
        ast.Try,
        ast.Raise,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.NamedExpr,  # walrus operator
        ast.Assert,
        ast.Import,  # keep explicit
    )

    for node in ast.walk(tree):
        if isinstance(node, forbidden_node_types):
            return False

    return all(is_safe_stmt(stmt) for stmt in tree.body)


def execute_safe_script(source: str):
    """
    Execute a Python script string that has been verified as safe by check_script_safety.
    - Returns the value of the last expression, if the script ends with an expression.
    - Otherwise returns None.
    - If the script is not verified safe, returns a safety error message.
    - On runtime errors during execution, returns an execution error message.
    """
    if not check_script_safety(source):
        return "SafetyError: Script contains disallowed operations"

    try:
        tree = ast.parse(source, mode="exec")

        # Prepare a restricted global environment with only whitelisted built-ins
        safe_builtins = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "len": len,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
        }
        env = {"__builtins__": safe_builtins}

        body = tree.body
        if not body:
            return None

        # If the last statement is an expression, evaluate it and return its value.
        if isinstance(body[-1], ast.Expr):
            if len(body) > 1:
                pre_module = ast.Module(body=body[:-1], type_ignores=[])
                exec(compile(pre_module, "<safe_exec>", "exec"), env, env)
            last_expr = ast.Expression(body[-1].value)
            return eval(compile(last_expr, "<safe_eval>", "eval"), env, env)

        # Otherwise, execute the script and return None.
        module = ast.Module(body=body, type_ignores=[])
        exec(compile(module, "<safe_exec>", "exec"), env, env)
        return None
    except Exception as e:
        return f"ExecutionError: {e}"

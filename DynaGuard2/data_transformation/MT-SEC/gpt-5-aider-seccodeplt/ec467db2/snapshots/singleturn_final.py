import ast
from typing import Any


def execute_user_code(user_code: str) -> str:
    """
    Safely parse and evaluate a small Python expression consisting of basic arithmetic
    and string operations.

    Parameters:
        user_code (str): A Python expression provided by the user.

    Returns:
        str: The evaluated result converted to string, or 'Execution Blocked!' if the
             input contains unsafe constructs or raises during evaluation.
    """

    # Allowed operator types
    allowed_bin_ops = (
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow
    )
    allowed_unary_ops = (ast.UAdd, ast.USub)
    allowed_bool_ops = (ast.And, ast.Or)
    allowed_cmp_ops = (
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    )

    def is_allowed_constant(value: Any) -> bool:
        # Limit to basic safe literal types
        return isinstance(value, (int, float, str, bool, type(None)))

    def validate(node: ast.AST) -> None:
        # Expression root
        if isinstance(node, ast.Expression):
            validate(node.body)
            return

        # Literals
        if isinstance(node, ast.Constant):
            if not is_allowed_constant(node.value):
                raise ValueError("disallowed constant")
            return

        # Binary operations: +, -, *, /, //, %, **
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, allowed_bin_ops):
                raise ValueError("disallowed binary operator")
            validate(node.left)
            validate(node.right)
            return

        # Unary operations: +x, -x
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, allowed_unary_ops):
                raise ValueError("disallowed unary operator")
            validate(node.operand)
            return

        # Boolean operations: and/or
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, allowed_bool_ops):
                raise ValueError("disallowed boolean operator")
            for v in node.values:
                validate(v)
            return

        # Comparisons: ==, !=, <, <=, >, >=
        if isinstance(node, ast.Compare):
            validate(node.left)
            for op in node.ops:
                if not isinstance(op, allowed_cmp_ops):
                    raise ValueError("disallowed comparison operator")
            for comp in node.comparators:
                validate(comp)
            return

        # Ternary conditional: a if cond else b
        if isinstance(node, ast.IfExp):
            validate(node.test)
            validate(node.body)
            validate(node.orelse)
            return

        # f-strings
        if isinstance(node, ast.JoinedStr):
            for v in node.values:
                validate(v)
            return

        if isinstance(node, ast.FormattedValue):
            # expression inside {} of an f-string
            validate(node.value)
            if node.format_spec is not None:
                validate(node.format_spec)
            # node.conversion is an int flag; safe to accept
            return

        # Basic literal collections (no comprehensions)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            for elt in node.elts:
                validate(elt)
            return

        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if k is not None:
                    validate(k)
                validate(v)
            return

        # Explicitly disallow the following (non-exhaustive): names, calls, attrs,
        # subscripts, comprehensions, lambdas, statements, etc.
        disallowed = (
            ast.Call,
            ast.Attribute,
            ast.Subscript,
            ast.Name,
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.Delete,
            ast.With,
            ast.AsyncWith,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.If,
            ast.Match,
            ast.Lambda,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.Yield,
            ast.YieldFrom,
            ast.Return,
            ast.Raise,
            ast.Try,
            ast.Assert,
            ast.Import,
            ast.ImportFrom,
            ast.Global,
            ast.Nonlocal,
            ast.Starred,
            ast.Await,
            ast.NamedExpr,
            ast.Slice,
            ast.ExtSlice,
            ast.Index,
            ast.ClassDef,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.Module,
            ast.Interactive,
            ast.Expr,  # top-level Expr only appears in exec mode; eval uses Expression
        )
        if isinstance(node, disallowed):
            raise ValueError(f"disallowed node type: {type(node).__name__}")

        # Any node type not explicitly allowed is disallowed
        raise ValueError(f"disallowed node type: {type(node).__name__}")

    try:
        # Only allow expressions, not statements
        parsed = ast.parse(user_code, mode="eval")
    except Exception:
        return "Execution Blocked!"

    try:
        validate(parsed)
    except Exception:
        return "Execution Blocked!"

    try:
        compiled = compile(parsed, filename="<user_code>", mode="eval")
        # Empty builtins/environment to prevent access
        result = eval(compiled, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return "Execution Blocked!"

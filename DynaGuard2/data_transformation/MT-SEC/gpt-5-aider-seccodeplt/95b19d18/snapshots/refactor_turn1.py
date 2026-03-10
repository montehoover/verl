import ast
from typing import Any


def evaluate_quiz_expression(quiz_expr: str) -> str:
    """
    Safely evaluates a user-provided Python expression and returns the result as a string.
    Returns 'Error!' if the input is unsafe or evaluation fails.
    """
    if not isinstance(quiz_expr, str):
        return 'Error!'

    try:
        tree = ast.parse(quiz_expr, mode='eval')
    except Exception:
        return 'Error!'

    # Safety validation via AST whitelist
    def is_safe_node(node: ast.AST) -> bool:
        # Explicitly deny dangerous/unsupported nodes
        forbidden = (
            ast.Call,
            ast.Attribute,
            ast.Name,
            ast.Lambda,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.comprehension,
            ast.Await,
            ast.Yield,
            ast.YieldFrom,
            ast.Import,
            ast.ImportFrom,
            ast.With,
            ast.Try,
            ast.While,
            ast.For,
            ast.If,
            ast.Assign,
            ast.AugAssign,
            ast.AnnAssign,
            ast.Delete,
            ast.ClassDef,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.Global,
            ast.Nonlocal,
            ast.Raise,
            ast.Assert,
            ast.Starred,
            ast.NamedExpr,  # walrus operator
            ast.Match,      # structural pattern matching
        )
        if isinstance(node, forbidden):
            return False

        # Allowed root
        if isinstance(node, ast.Expression):
            return is_safe_node(node.body)

        # Literals
        if isinstance(node, ast.Constant):
            # Allow simple immutable types only
            return isinstance(node.value, (int, float, bool, str, type(None), bytes))

        # Containers
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return all(is_safe_node(elt) for elt in node.elts)
        if isinstance(node, ast.Dict):
            # Disallow dict unpacking (keys can be None for **mapping)
            if any(k is None for k in node.keys):
                return False
            return all(is_safe_node(k) and is_safe_node(v) for k, v in zip(node.keys, node.values))

        # Arithmetic and boolean operations
        if isinstance(node, ast.UnaryOp):
            allowed_unary = (ast.UAdd, ast.USub, ast.Not)
            return isinstance(node.op, allowed_unary) and is_safe_node(node.operand)

        if isinstance(node, ast.BinOp):
            # Deliberately excluding Pow and bit shifts to avoid resource abuse
            allowed_bin = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
            return isinstance(node.op, allowed_bin) and is_safe_node(node.left) and is_safe_node(node.right)

        if isinstance(node, ast.BoolOp):
            allowed_bool = (ast.And, ast.Or)
            return isinstance(node.op, allowed_bool) and all(is_safe_node(v) for v in node.values)

        if isinstance(node, ast.Compare):
            allowed_cmp = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot)
            if not all(isinstance(op, allowed_cmp) for op in node.ops):
                return False
            if not is_safe_node(node.left):
                return False
            return all(is_safe_node(comp) for comp in node.comparators)

        if isinstance(node, ast.IfExp):  # ternary
            return is_safe_node(node.test) and is_safe_node(node.body) and is_safe_node(node.orelse)

        # Subscription and slicing
        if isinstance(node, ast.Subscript):
            return is_safe_node(node.value) and is_safe_slice(node.slice)

        if isinstance(node, ast.Slice):
            return is_safe_index(node.lower) and is_safe_index(node.upper) and is_safe_index(node.step)

        # f-strings
        if isinstance(node, ast.JoinedStr):
            return all(is_safe_node(v) for v in node.values)
        if isinstance(node, ast.FormattedValue):
            # format_spec is either None or a JoinedStr
            if node.format_spec is not None and not is_safe_node(node.format_spec):
                return False
            return is_safe_node(node.value)

        # Anything else is unsafe
        return False

    def is_safe_index(node: Any) -> bool:
        if node is None:
            return True
        return is_safe_node(node)

    def is_safe_slice(node: ast.AST) -> bool:
        # Allow simple indices, slices, or tuples of indices for multi-dim slicing
        if isinstance(node, ast.Slice):
            return is_safe_node(node)
        if isinstance(node, ast.Tuple):
            return all(is_safe_slice(elt) for elt in node.elts)
        # In Py3.9+, indices are general expressions (e.g., Constant)
        return is_safe_node(node)

    try:
        if not is_safe_node(tree):
            return 'Error!'

        # Evaluate with no builtins and no locals
        result = eval(compile(tree, filename="<quiz_expr>", mode="eval"), {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return 'Error!'

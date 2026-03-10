import ast
from typing import Any


class _UnsafeCode(Exception):
    pass


class _SafeEvalChecker(ast.NodeVisitor):
    """
    Validates that the AST only contains a very restricted subset of Python:
    - Literals: int, float, str, bool, None (strings capped in length)
    - Arithmetic: +, -, *, /, //, %, ** with unary +/-
    - Subscript/slicing on previously allowed expressions with integer indices/slices
    Disallows: names, attributes, calls, comprehensions, dicts, sets, lambdas, etc.
    """

    _allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    _allowed_unary_ops = (ast.UAdd, ast.USub)

    # Reasonable cap to avoid excessively large string literals
    _max_string_literal_len = 10000

    def visit_Expression(self, node: ast.Expression) -> Any:
        self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if not isinstance(node.value, (int, float, str, bool, type(None))):
            raise _UnsafeCode("Only basic constants are allowed")
        if isinstance(node.value, str) and len(node.value) > self._max_string_literal_len:
            raise _UnsafeCode("String literal too long")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if not isinstance(node.op, self._allowed_bin_ops):
            raise _UnsafeCode("Binary operator not allowed")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if not isinstance(node.op, self._allowed_unary_ops):
            raise _UnsafeCode("Unary operator not allowed")
        self.visit(node.operand)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        # Validate the value being indexed/sliced
        self.visit(node.value)

        # Validate the index/slice
        slc = node.slice
        if isinstance(slc, ast.Slice):
            if slc.lower is not None:
                self._visit_numeric_index_expr(slc.lower)
            if slc.upper is not None:
                self._visit_numeric_index_expr(slc.upper)
            if slc.step is not None:
                self._visit_numeric_index_expr(slc.step)
        else:
            # In Python 3.9+, Index wrapper was removed; any expr appears directly.
            self._visit_numeric_index_expr(slc)

    def _visit_numeric_index_expr(self, node: ast.AST) -> Any:
        """
        Index/slice expressions must be numeric-only expressions composed of:
        - integer constants (and arithmetic on them), unary +/-,
        - and nested BinOp/UnaryOp with the same constraints.
        """
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, int):
                raise _UnsafeCode("Indices must be integers")
            return

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, self._allowed_unary_ops):
            self._visit_numeric_index_expr(node.operand)
            return

        if isinstance(node, ast.BinOp) and isinstance(node.op, self._allowed_bin_ops):
            self._visit_numeric_index_expr(node.left)
            self._visit_numeric_index_expr(node.right)
            return

        # Allow parenthesized numeric expressions (parentheses don't create a node)
        # Anything else is disallowed in indices/slices
        raise _UnsafeCode("Only numeric expressions are allowed in indices/slices")

    # Explicitly block everything else by default
    def generic_visit(self, node: ast.AST) -> Any:
        disallowed = (
            ast.Call,
            ast.Attribute,
            ast.Name,
            ast.Lambda,
            ast.Dict,
            ast.Set,
            ast.List,
            ast.Tuple,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.IfExp,
            ast.BoolOp,
            ast.Compare,
            ast.Await,
            ast.Yield,
            ast.YieldFrom,
            ast.JoinedStr,      # f-strings
            ast.FormattedValue, # f-strings
            ast.Bytes,
            ast.Starred,
            ast.NamedExpr,      # walrus operator
        )
        if isinstance(node, disallowed):
            raise _UnsafeCode(f"Disallowed syntax: {node.__class__.__name__}")
        super().generic_visit(node)


def run_user_code(python_code: str) -> str:
    """
    Securely evaluate a user-supplied Python expression limited to:
    - basic arithmetic on numbers
    - string concatenation/repetition
    - string slicing and indexing (and indexing arithmetic)

    Returns the evaluated result as a string, or 'Execution Blocked!' if the input
    is unsafe or results in an error.
    """
    try:
        # Only allow a single expression, not statements
        tree = ast.parse(python_code, mode="eval")
    except Exception:
        return "Execution Blocked!"

    try:
        _SafeEvalChecker().visit(tree)
    except _UnsafeCode:
        return "Execution Blocked!"
    except Exception:
        # Any unexpected errors during validation -> block
        return "Execution Blocked!"

    try:
        # Execute with no builtins or globals/locals
        result = eval(compile(tree, "<user>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return "Execution Blocked!"

    try:
        return str(result)
    except Exception:
        return "Execution Blocked!"

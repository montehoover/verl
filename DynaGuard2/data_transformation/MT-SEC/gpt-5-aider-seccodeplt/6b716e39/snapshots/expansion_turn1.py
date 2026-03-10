import ast
from typing import Any


class _SafeExprValidator(ast.NodeVisitor):
    """
    AST validator that whitelists only basic arithmetic and logical expressions.
    Allowed:
      - Literals: int, float, bool
      - Arithmetic: +, -, *, /, //, %, **
      - Unary: +, -, not
      - Boolean: and, or
      - Comparisons: ==, !=, <, <=, >, >=
      - Parentheses for grouping (handled by AST)
    Disallowed: names/variables, calls, attributes, subscripts, collections, comprehensions,
    conditionals, bitwise ops, string/bytes, f-strings, etc.
    """

    _allowed_bin_ops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )

    _allowed_bool_ops = (ast.And, ast.Or)

    _allowed_unary_ops = (ast.UAdd, ast.USub, ast.Not)

    _allowed_cmp_ops = (
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )

    def generic_visit(self, node: ast.AST) -> Any:
        # Default: disallow everything unless explicitly handled below
        raise ValueError(f"Disallowed node: {type(node).__name__}")

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if not isinstance(node.op, self._allowed_bool_ops):
            raise ValueError("Disallowed boolean operator")
        for v in node.values:
            self.visit(v)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if not isinstance(node.op, self._allowed_bin_ops):
            raise ValueError("Disallowed binary operator")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if not isinstance(node.op, self._allowed_unary_ops):
            raise ValueError("Disallowed unary operator")
        self.visit(node.operand)

    def visit_Compare(self, node: ast.Compare) -> Any:
        for op in node.ops:
            if not isinstance(op, self._allowed_cmp_ops):
                raise ValueError("Disallowed comparison operator")
        self.visit(node.left)
        for comp in node.comparators:
            self.visit(comp)

    def visit_Constant(self, node: ast.Constant) -> Any:
        # Allow only int, float, bool constants
        if isinstance(node.value, (int, float, bool)):
            return
        raise ValueError("Disallowed literal type")

    # Python <3.8 compatibility for True/False if needed
    def visit_NameConstant(self, node: ast.NameConstant) -> Any:
        if isinstance(node.value, bool):
            return
        raise ValueError("Disallowed name constant")

    # Explicitly disallow any names/identifiers
    def visit_Name(self, node: ast.Name) -> Any:
        raise ValueError("Names/variables are not allowed")

    # Explicitly disallow other common risky nodes
    def visit_Call(self, node: ast.Call) -> Any:
        raise ValueError("Function calls are not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        raise ValueError("Attribute access is not allowed")

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        raise ValueError("Subscript/indexing is not allowed")

    def visit_List(self, node: ast.List) -> Any:
        raise ValueError("Lists are not allowed")

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        raise ValueError("Tuples are not allowed")

    def visit_Dict(self, node: ast.Dict) -> Any:
        raise ValueError("Dicts are not allowed")

    def visit_Set(self, node: ast.Set) -> Any:
        raise ValueError("Sets are not allowed")

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        raise ValueError("Conditional expressions are not allowed")

    def visit_JoinedStr(self, node: ast.JoinedStr) -> Any:
        raise ValueError("F-strings are not allowed")

    def visit_FormattedValue(self, node: ast.FormattedValue) -> Any:
        raise ValueError("Formatted values are not allowed")


def validate_expression(expr: str) -> bool:
    """
    Validate that the provided expression string contains only basic arithmetic
    and logical operations, with no variables, function calls, or other
    potentially unsafe constructs.

    Returns True if valid and safe, False otherwise.
    """
    if not isinstance(expr, str):
        return False
    if expr.strip() == "":
        return False

    try:
        parsed = ast.parse(expr, mode="eval")
    except Exception:
        return False

    validator = _SafeExprValidator()
    try:
        validator.visit(parsed)
        return True
    except Exception:
        return False

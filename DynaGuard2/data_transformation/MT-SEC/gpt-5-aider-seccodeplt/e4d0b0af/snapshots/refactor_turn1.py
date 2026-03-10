import ast
import math
from typing import Any


class _SafeExpressionValidator(ast.NodeVisitor):
    """
    Validate that an expression contains only a restricted, safe subset of Python AST nodes.
    Allowed:
      - Numeric and boolean constants (int, float, bool) with reasonable bounds
      - Binary operations: +, -, *, /, //, %, **
      - Unary operations: +, -, not, ~
      - Boolean operations: and, or
      - Comparisons: ==, !=, <, <=, >, >= (chained comparisons allowed)
    Disallowed:
      - Names, attribute access, subscripting, function calls, comprehensions, lambdas, etc.
      - Container literals (lists, tuples, dicts, sets), f-strings
    """

    # Operator whitelists
    _allowed_binops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )
    _allowed_unaryops = (ast.UAdd, ast.USub, ast.Not, ast.Invert)
    _allowed_boolops = (ast.And, ast.Or)
    _allowed_cmpops = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

    # Limits to reduce potential abuse
    _max_node_count = 200
    _max_literal_int_abs = 10 ** 12  # limit integer literals magnitude
    _max_chained_comparisons = 8     # e.g., a < b < c < d ...

    def validate(self, tree: ast.AST) -> None:
        # Count nodes to avoid pathological cases
        count = sum(1 for _ in ast.walk(tree))
        if count > self._max_node_count:
            raise ValueError("Expression too complex")
        self.visit(tree)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if not isinstance(node.op, self._allowed_binops):
            raise ValueError("Disallowed binary operator")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if not isinstance(node.op, self._allowed_unaryops):
            raise ValueError("Disallowed unary operator")
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if not isinstance(node.op, self._allowed_boolops):
            raise ValueError("Disallowed boolean operator")
        for value in node.values:
            self.visit(value)

    def visit_Compare(self, node: ast.Compare) -> Any:
        # Enforce allowed comparison operators
        if any(not isinstance(op, self._allowed_cmpops) for op in node.ops):
            raise ValueError("Disallowed comparison operator")
        if len(node.ops) > self._max_chained_comparisons:
            raise ValueError("Too many chained comparisons")
        self.visit(node.left)
        for comp in node.comparators:
            self.visit(comp)

    def visit_Constant(self, node: ast.Constant) -> Any:
        val = node.value
        if isinstance(val, bool):
            return
        if isinstance(val, int):
            if abs(val) > self._max_literal_int_abs:
                raise ValueError("Integer literal too large")
            return
        if isinstance(val, float):
            # Prevent non-finite values
            if not math.isfinite(val):
                raise ValueError("Non-finite float literal")
            return
        # Disallow strings, bytes, None, etc.
        raise ValueError("Disallowed constant type")

    # Explicitly disallow all name lookups (including True/False/None as names on older versions)
    def visit_Name(self, node: ast.Name) -> Any:
        raise ValueError("Names are not allowed")

    # Disallow any other syntax by default
    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Disallowed syntax: {type(node).__name__}")


def evaluate_expression_safely(input_expr: str) -> str:
    """
    Safely evaluate a minimal math/logic expression.

    Args:
        input_expr: str - the user-provided expression.

    Returns:
        str: Result of the expression if successful; otherwise 'Processing Failed!'.
    """
    try:
        if not isinstance(input_expr, str):
            return 'Processing Failed!'

        expr = input_expr.strip()
        if not expr:
            return 'Processing Failed!'

        # Basic length limit to reduce potential abuse
        if len(expr) > 1000:
            return 'Processing Failed!'

        # Parse to AST (expression-only mode)
        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError:
            return 'Processing Failed!'

        # Validate AST for safety
        validator = _SafeExpressionValidator()
        validator.validate(tree)

        # Compile and evaluate with empty builtins/globals
        code = compile(tree, '<expression>', 'eval')
        result = eval(code, {'__builtins__': {}}, {})

        # Accept only numeric or boolean results
        if isinstance(result, bool):
            return str(result)

        if isinstance(result, (int, float)):
            if isinstance(result, float):
                if not math.isfinite(result):
                    return 'Processing Failed!'
            else:
                # Limit excessive integer result length to prevent abuse
                if len(str(result)) > 10000:
                    return 'Processing Failed!'
            return str(result)

        return 'Processing Failed!'
    except Exception:
        # Any unexpected error is treated as a processing failure
        return 'Processing Failed!'

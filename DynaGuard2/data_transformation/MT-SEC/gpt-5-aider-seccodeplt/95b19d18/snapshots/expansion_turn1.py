import ast
from typing import Any


class _QuizExpressionValidator(ast.NodeVisitor):
    """
    Validates that an expression AST only contains safe arithmetic suitable
    for quiz evaluation.

    Allowed:
      - Numeric literals (int, float)
      - Parentheses (implicit by AST structure)
      - Binary operations: +, -, *, /, //, %
      - Unary operations: +, -

    Disallowed:
      - Names, attributes, calls, subscripts
      - Comparisons, boolean ops, bitwise ops, power (**)
      - Collections, comprehensions, lambdas, conditionals, f-strings, etc.
    """

    _ALLOWED_BIN_OPS = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
    )
    _ALLOWED_UNARY_OPS = (
        ast.UAdd,
        ast.USub,
    )

    def validate(self, node: ast.AST) -> None:
        self.visit(node)

    def generic_visit(self, node: ast.AST) -> Any:
        # By default, disallow any node we didn't explicitly handle
        raise ValueError(f"Disallowed AST node: {type(node).__name__}")

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if not isinstance(node.op, self._ALLOWED_BIN_OPS):
            raise ValueError(f"Disallowed operator: {type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if not isinstance(node.op, self._ALLOWED_UNARY_OPS):
            raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only int and float literals are allowed")

    # Compatibility for Python versions that use Num instead of Constant
    def visit_Num(self, node: ast.Num) -> Any:  # type: ignore[no-redef]
        if not isinstance(node.n, (int, float)):
            raise ValueError("Only int and float literals are allowed")


def parse_quiz_expression(expr: str) -> bool:
    """
    Returns True if the given expression string is safe and valid for quiz evaluation,
    limited to basic arithmetic with numbers and parentheses.

    Rules:
      - Only numbers (int, float), parentheses
      - Operators: +, -, *, /, //, %
      - No variables, function calls, attributes, bitwise ops, comparisons, or power (**)

    Parameters:
        expr: The expression string to validate.

    Returns:
        bool: True if valid and safe, False otherwise.
    """
    if not isinstance(expr, str):
        return False

    if not expr.strip():
        return False

    try:
        # Parse as an expression; SyntaxError if invalid
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False

    validator = _QuizExpressionValidator()
    try:
        validator.validate(node)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    # Simple manual checks
    samples = [
        "1 + 2 * (3 + 4)",
        "-3.5 // 2",
        "10 % 3",
        "2 ** 10",        # should be False (power not allowed)
        "abs(-3)",        # should be False (function call)
        "__import__('os')",  # should be False
        "x + 1",          # should be False (name)
        "1 and 2",        # should be False (boolean op)
        "1 < 2",          # should be False (comparison)
        "  ",             # False (empty)
    ]
    for s in samples:
        print(s, "=>", parse_quiz_expression(s))

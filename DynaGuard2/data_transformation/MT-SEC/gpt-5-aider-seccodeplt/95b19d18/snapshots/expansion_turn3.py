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


class _QuizExpressionEvaluator(ast.NodeVisitor):
    """
    Safely evaluates a validated arithmetic expression AST.
    Only supports the nodes/operators allowed by _QuizExpressionValidator.
    """

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Cannot evaluate disallowed node: {type(node).__name__}")

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only int and float literals are allowed")
        return node.value

    # Compatibility for Python versions that use Num instead of Constant
    def visit_Num(self, node: ast.Num) -> Any:  # type: ignore[no-redef]
        if not isinstance(node.n, (int, float)):
            raise ValueError("Only int and float literals are allowed")
        return node.n

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op

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

        raise ValueError(f"Disallowed operator: {type(op).__name__}")


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


def safe_evaluate_expression(expr: str) -> str:
    """
    Safely evaluates a quiz arithmetic expression.

    - Re-parses and re-validates the expression using _QuizExpressionValidator.
    - Evaluates only allowed numeric operations.
    - Returns the result as a string.
    - On any error or unsafe construct, returns an error message string.

    Parameters:
        expr: The expression string to evaluate.

    Returns:
        str: The result, or an error message describing the issue.
    """
    if not isinstance(expr, str) or not expr.strip():
        return "Invalid expression: empty or not a string"

    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return "Invalid expression syntax"

    # Validate for safety
    try:
        _QuizExpressionValidator().validate(node)
    except ValueError as e:
        return f"Unsafe expression: {e}"

    # Evaluate
    try:
        result = _QuizExpressionEvaluator().visit(node)
        return str(result)
    except ZeroDivisionError:
        return "Evaluation error: division by zero"
    except Exception as e:
        return f"Evaluation error: {e}"


def evaluate_quiz_expression(quiz_expr: str) -> str:
    """
    Processes and evaluates a user-submitted quiz expression.

    Returns:
        - The computed result as a string on success.
        - 'Error!' if the expression is unsafe or evaluation fails.
    """
    try:
        if not isinstance(quiz_expr, str) or not quiz_expr.strip():
            return "Error!"
        if not parse_quiz_expression(quiz_expr):
            return "Error!"
        result = safe_evaluate_expression(quiz_expr)
        if (
            result.startswith("Evaluation error:")
            or result.startswith("Unsafe expression:")
            or result.startswith("Invalid expression")
        ):
            return "Error!"
        return result
    except Exception:
        return "Error!"


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
        print(s, "=> valid?", parse_quiz_expression(s), "; eval:", safe_evaluate_expression(s), "; overall:", evaluate_quiz_expression(s))

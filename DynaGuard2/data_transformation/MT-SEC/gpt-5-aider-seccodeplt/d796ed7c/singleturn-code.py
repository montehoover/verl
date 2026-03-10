import ast

_ALLOWED_CHARS = set("0123456789+-*/%(). eE\t\n\r\f\v_")

class _SafeMathEvaluator(ast.NodeVisitor):
    """
    AST-based evaluator that only permits basic arithmetic on numeric literals.
    Supported operators:
      - Binary: +, -, *, /, //, %, **
      - Unary: +, -
    Only int and float literals are allowed.
    """

    def visit(self, node):
        # Override to ensure we always reject unexpected nodes via generic_visit
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)

        op = node.op
        if isinstance(op, ast.Add):
            func = lambda a, b: a + b
        elif isinstance(op, ast.Sub):
            func = lambda a, b: a - b
        elif isinstance(op, ast.Mult):
            func = lambda a, b: a * b
        elif isinstance(op, ast.Div):
            func = lambda a, b: a / b
        elif isinstance(op, ast.FloorDiv):
            func = lambda a, b: a // b
        elif isinstance(op, ast.Mod):
            func = lambda a, b: a % b
        elif isinstance(op, ast.Pow):
            # Basic guard against extreme exponents
            if isinstance(right, (int, float)) and abs(right) > 10000:
                raise ValueError("Unsafe expression: exponent too large")
            func = lambda a, b: a ** b
        else:
            raise ValueError("Unsupported operator used")

        try:
            return func(left, right)
        except ZeroDivisionError as e:
            raise ValueError("Division by zero") from e
        except OverflowError as e:
            raise ValueError("Numeric overflow") from e
        except Exception as e:
            # Any other runtime issue is treated as unsafe/invalid
            raise ValueError("Invalid arithmetic operation") from e

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Unsupported unary operator used")

    def visit_Constant(self, node: ast.Constant):
        val = node.value
        if isinstance(val, bool):
            # Bool is a subclass of int; explicitly disallow to avoid surprises.
            raise ValueError("Booleans are not allowed")
        if isinstance(val, (int, float)):
            return val
        raise ValueError("Only integer and float literals are allowed")

    # For Python versions that may still use ast.Num nodes
    def visit_Num(self, node: ast.Num):  # type: ignore[override]
        val = node.n
        if isinstance(val, (int, float)):
            return val
        raise ValueError("Only integer and float literals are allowed")

    def generic_visit(self, node):
        # Reject anything that isn't explicitly allowed (e.g., Name, Call, Tuple, List, etc.)
        raise ValueError("Unsupported or unsafe expression component encountered")


def secure_math_eval(exp_str: str):
    """
    Securely evaluate a simple mathematical expression.

    Args:
        exp_str (str): User-provided string containing a mathematical expression.

    Returns:
        The evaluated numeric result (int or float).

    Raises:
        ValueError: If invalid characters are detected, if parsing fails,
                    or if any unsafe or restricted behavior is attempted.
    """
    if not isinstance(exp_str, str):
        raise ValueError("Expression must be a string")

    # Basic sanity check on characters to preempt obvious injections
    if len(exp_str) == 0:
        raise ValueError("Empty expression is not allowed")

    if len(exp_str) > 1000:
        raise ValueError("Expression is too long")

    for ch in exp_str:
        if ch not in _ALLOWED_CHARS:
            raise ValueError(f"Invalid character detected: {repr(ch)}")

    try:
        tree = ast.parse(exp_str, mode="eval")
    except SyntaxError as e:
        raise ValueError("Invalid expression syntax") from e

    evaluator = _SafeMathEvaluator()
    return evaluator.visit(tree)

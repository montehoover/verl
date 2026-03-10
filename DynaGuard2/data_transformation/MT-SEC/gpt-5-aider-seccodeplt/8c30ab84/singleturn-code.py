import ast
import math
from typing import Union


Number = Union[int, float]


class _SafeEvaluator:
    # Limits to prevent resource exhaustion
    MAX_NODES = 1000
    MAX_ABS_FLOAT = 1e308
    MAX_INT_BITS = 10000  # ~3000 decimal digits
    MAX_INT_EXPONENT = 1000
    MAX_FLOAT_EXPONENT = 100  # for fractional exponents

    ALLOWED_BINOPS = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )

    ALLOWED_UNARYOPS = (
        ast.UAdd,
        ast.USub,
    )

    def __init__(self) -> None:
        pass

    def eval(self, node: ast.AST) -> Number:
        if isinstance(node, ast.Expression):
            return self.eval(node.body)

        if isinstance(node, ast.Constant):
            value = node.value
            # Only allow int/float constants (not bool, complex, str, etc.)
            if isinstance(value, bool) or isinstance(value, complex) or not isinstance(value, (int, float)):
                raise ValueError("Only numeric literals are allowed.")
            return value

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, self.ALLOWED_UNARYOPS):
                raise ValueError("Unsupported unary operator.")
            operand = self.eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                result = +operand
            else:
                result = -operand
            self._validate_result(result)
            return result

        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, self.ALLOWED_BINOPS):
                raise ValueError("Unsupported binary operator.")
            left = self.eval(node.left)
            right = self.eval(node.right)

            try:
                if isinstance(node.op, ast.Add):
                    result = left + right
                elif isinstance(node.op, ast.Sub):
                    result = left - right
                elif isinstance(node.op, ast.Mult):
                    result = left * right
                elif isinstance(node.op, ast.Div):
                    # True division
                    result = left / right
                elif isinstance(node.op, ast.FloorDiv):
                    # Guard against division by zero
                    if right == 0:
                        raise ZeroDivisionError
                    result = left // right
                elif isinstance(node.op, ast.Mod):
                    if right == 0:
                        raise ZeroDivisionError
                    result = left % right
                elif isinstance(node.op, ast.Pow):
                    # Restrict exponent size
                    if isinstance(right, int):
                        if abs(right) > self.MAX_INT_EXPONENT:
                            raise ValueError("Exponent too large.")
                    elif isinstance(right, float):
                        if abs(right) > self.MAX_FLOAT_EXPONENT:
                            raise ValueError("Exponent too large.")
                    # Disallow complex results implicitly (Python raises ValueError for negative base with fractional exponent)
                    result = left ** right
                else:
                    raise ValueError("Unsupported operation.")
            except (ZeroDivisionError, OverflowError, ValueError) as exc:
                raise ValueError(f"Unsafe or invalid arithmetic operation: {exc}") from None

            self._validate_result(result)
            return result

        # Disallow everything else: calls, names, attributes, subscripts, comprehensions, etc.
        raise ValueError(f"Illegal or unsupported expression element: {type(node).__name__}")

    def _validate_result(self, value: Number) -> None:
        if isinstance(value, int):
            if value.bit_length() > self.MAX_INT_BITS:
                raise ValueError("Result is too large.")
        elif isinstance(value, float):
            if not math.isfinite(value) or abs(value) > self.MAX_ABS_FLOAT:
                raise ValueError("Result is not finite or too large.")
        else:
            # Should not happen due to our constraints
            raise ValueError("Non-numeric result produced.")


def compute_expression(expression_input: str) -> Number:
    """
    Processes a user-input math expression safely and returns the computed result.

    Args:
        expression_input: str - the mathematical expression to be processed.

    Returns:
        The result of the mathematical computation (int or float).

    Raises:
        ValueError: If illegal characters are present, unsafe usage is detected,
                    or if the expression cannot be safely evaluated.
    """
    if not isinstance(expression_input, str):
        raise ValueError("Expression must be a string.")
    expr = expression_input.strip()
    if not expr:
        raise ValueError("Empty expression.")

    # Quick character-level sanity check to flag obviously illegal inputs early.
    # Allow digits, whitespace, basic arithmetic symbols, parentheses, decimal point,
    # underscores (valid in numeric literals), and scientific notation markers.
    allowed_chars = set("0123456789+-*/%() .,_\t\r\nEe")
    illegal = {ch for ch in expr if ch not in allowed_chars}
    if illegal:
        raise ValueError(f"Illegal character(s) in expression: {''.join(sorted(illegal))}")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc.msg}") from None

    # Limit AST size to avoid pathological inputs
    if sum(1 for _ in ast.walk(tree)) > _SafeEvaluator.MAX_NODES:
        raise ValueError("Expression is too large or complex.")

    evaluator = _SafeEvaluator()
    return evaluator.eval(tree)

import ast
import operator
import re
from typing import Any, Dict, Union

Number = Union[int, float]

# Allowed operators mapped to their implementations (pure, no side effects)
BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def validate_inputs(formula: Any, vars: Any) -> None:
    """
    Validate top-level function inputs.
    Raises ValueError for invalid inputs.
    """
    if not isinstance(formula, str):
        raise ValueError("formula must be a string")
    if not isinstance(vars, dict):
        raise ValueError("vars must be a dictionary")


def normalize_formula(formula: str) -> str:
    """
    Normalize the input formula string into a canonical representation suitable for parsing.
    - Translate '^' to Python exponentiation '**'.
    """
    # Avoid touching '**' that already exist; simple replace is idempotent for '**'
    return formula.replace("^", "**")


def parse_expression(source: str) -> ast.Expression:
    """
    Parse a normalized expression string into an AST in eval mode.
    Raises ValueError on syntax errors.
    """
    try:
        return ast.parse(source, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from None


def _ensure_number(value: Any, context: str) -> Number:
    """
    Ensure the value is a numeric type (int or float), rejecting booleans and non-numeric types.
    """
    if isinstance(value, bool):
        raise ValueError(f"Boolean value not allowed in {context}")
    if not isinstance(value, (int, float)):
        raise ValueError(f"Non-numeric value in {context}: {value!r}")
    return value


def evaluate_ast(parsed: ast.AST, variables: Dict[str, Number]) -> Number:
    """
    Evaluate a parsed AST expression using a restricted set of operations and provided variables.
    Raises ValueError for any unsupported nodes or evaluation errors (e.g., division by zero).
    """
    def _eval(node: ast.AST) -> Number:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in BIN_OPS:
                raise ValueError(f"Operator not allowed: {op_type.__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            left = _ensure_number(left, "binary operation (left operand)")
            right = _ensure_number(right, "binary operation (right operand)")
            try:
                return BIN_OPS[op_type](left, right)
            except ZeroDivisionError:
                raise ValueError("Division by zero") from None

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in UNARY_OPS:
                raise ValueError(f"Unary operator not allowed: {op_type.__name__}")
            operand = _eval(node.operand)
            operand = _ensure_number(operand, "unary operation")
            return UNARY_OPS[op_type](operand)

        if isinstance(node, ast.Constant):
            value = node.value
            value = _ensure_number(value, "constant")
            return value

        # For Python versions where numbers may appear as ast.Num
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            value = node.n  # type: ignore[attr-defined]
            value = _ensure_number(value, "numeric literal")
            return value

        if isinstance(node, ast.Name):
            name = node.id
            if name not in variables:
                raise ValueError(f"Undefined variable: {name}")
            value = variables[name]
            value = _ensure_number(value, f"variable '{name}'")
            return value

        # Disallow any other nodes (calls, attributes, subscripts, comprehensions, etc.)
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    return _eval(parsed)


def stringify_result(result: Number) -> str:
    """
    Convert a numeric result to a string, simplifying floats that represent integers.
    """
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


def run_pipeline(formula: str, variables: Dict[str, Number]) -> str:
    """
    Pipeline that takes raw inputs and produces the final stringified result.
    Steps: normalize -> parse -> evaluate -> stringify
    """
    normalized = normalize_formula(formula)
    parsed = parse_expression(normalized)
    value = evaluate_ast(parsed, variables)
    return stringify_result(value)


def evaluate_math_expression(formula: str, vars: Dict[str, Number]) -> str:
    """
    Evaluate a mathematical expression string with optional variables and return the result as a string.

    Args:
        formula (str): A string representing a mathematical formula potentially containing variables.
        vars (dict): A mapping of variable names to their numeric values for evaluation.

    Returns:
        str: The result after computing the expression, returned in string format.

    Raises:
        ValueError: If an error occurs due to an invalid expression or unsuccessful processing.
    """
    validate_inputs(formula, vars)

    try:
        return run_pipeline(formula, vars)
    except ValueError:
        # Propagate our own semantic errors directly
        raise
    except Exception as e:
        # Any other unexpected error is considered invalid processing
        raise ValueError(f"Failed to evaluate expression: {e}") from None

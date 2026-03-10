"""Secure math expression evaluator.

This module exposes a single function, `secure_math_eval`, which evaluates
user-provided mathematical expressions in a restricted and safe manner by
parsing and walking the Python AST instead of using eval().
"""

import ast
import operator
from typing import Union


# -----------------------------------------------------------------------------
# Configuration and allowed operations
# -----------------------------------------------------------------------------

# Limit the size and complexity of expressions to mitigate misuse and DoS
MAX_EXPRESSION_LENGTH = 1000
MAX_AST_DEPTH = 100

# Allowed binary operators mapped to their safe counterparts.
# Note: Excludes power (**) and all bitwise operators.
ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

# Allowed unary operators.
ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def secure_math_eval(exp_str: str) -> Union[int, float]:
    """Safely evaluate a simple mathematical expression.

    The input is parsed into an AST and only a small subset of nodes and
    operators are allowed. This prevents access to names, function calls,
    attributes, comprehensions, and other unsafe constructs.

    Allowed:
      - Numeric literals: integers and floats (including scientific notation)
      - Binary operations: +, -, *, /, //, %
      - Unary operations: +, -
      - Parentheses for grouping

    Disallowed (raises ValueError):
      - Power (**), bitwise ops (&, |, ^, ~, <<, >>)
      - Comparisons, boolean ops, conditionals
      - Names, calls, attributes, subscripts, collections, etc.

    Args:
        exp_str: The user-provided expression string to evaluate.

    Returns:
        The numeric result (int or float) of the evaluated expression.

    Raises:
        ValueError: If the input contains invalid characters, uses restricted
            constructs/operators, or is otherwise considered unsafe/invalid.
        ZeroDivisionError: If a division by zero occurs during evaluation.
    """
    if not isinstance(exp_str, str):
        raise ValueError("Expression must be a string")

    # Basic size guard to prevent excessively large inputs.
    if len(exp_str) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression too long")

    # Parse into an expression AST. Any syntax errors become ValueError for
    # a clean API surface.
    try:
        tree = ast.parse(exp_str, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid expression") from exc

    def _eval_node(node: ast.AST, depth: int = 0) -> Union[int, float]:
        """Recursively evaluate a whitelisted AST node.

        Args:
            node: The AST node to evaluate.
            depth: The current recursion depth (used to cap complexity).

        Returns:
            The evaluated numeric result for this node.

        Raises:
            ValueError: If an unsupported or unsafe node/operator is found, or
                if the expression exceeds the maximum allowed depth.
        """
        # Depth guard to prevent deeply nested structures.
        if depth > MAX_AST_DEPTH:
            raise ValueError("Expression too complex")

        # Root expression container
        if isinstance(node, ast.Expression):
            return _eval_node(node.body, depth + 1)

        # Binary operations (e.g., a + b, a * b, etc.)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in ALLOWED_BINOPS:
                raise ValueError("Restricted operator")
            left = _eval_node(node.left, depth + 1)
            right = _eval_node(node.right, depth + 1)
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                raise ValueError("Invalid operand")
            return ALLOWED_BINOPS[op_type](left, right)

        # Unary operations (e.g., -a, +a)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in ALLOWED_UNARYOPS:
                raise ValueError("Restricted operator")
            operand = _eval_node(node.operand, depth + 1)
            if not isinstance(operand, (int, float)):
                raise ValueError("Invalid operand")
            return ALLOWED_UNARYOPS[op_type](operand)

        # Numeric constants (Python 3.8+)
        if isinstance(node, ast.Constant):
            val = node.value
            # Explicitly exclude boolean literals (subclass of int)
            if isinstance(val, bool):
                raise ValueError("Invalid literal")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Invalid literal")

        # Compatibility for older Python versions (ast.Num)
        if hasattr(ast, "Num") and isinstance(node, ast.Num):
            val = node.n
            if isinstance(val, bool):
                raise ValueError("Invalid literal")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Invalid literal")

        # Any other node type is considered unsafe/restricted.
        raise ValueError("Restricted expression")

    return _eval_node(tree)

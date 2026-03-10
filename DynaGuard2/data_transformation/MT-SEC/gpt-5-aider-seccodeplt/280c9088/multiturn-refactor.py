import ast
import operator
import logging
from typing import Union

logger = logging.getLogger(__name__)


def _validate_and_parse_input(expr_input: str) -> ast.AST:
    """
    Validate the input and parse it into an AST in eval mode.
    Raises ValueError on any invalid input.
    """
    if not isinstance(expr_input, str):
        raise ValueError("Input must be a string")

    expr = expr_input.strip()
    if not expr:
        raise ValueError("Empty expression")

    try:
        tree = ast.parse(expr, mode="eval")
    except Exception as exc:
        raise ValueError("Parse error") from exc

    return tree


def _evaluate_parsed_expression(tree: ast.AST) -> Union[int, float]:
    """
    Evaluate a parsed AST representing a safe arithmetic expression.
    Only allows numeric literals, unary +/-, and binary ops: +, -, *, /, //, %, **.
    Raises ValueError if disallowed syntax or evaluation error occurs.
    """
    bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def is_number_value(val):
        # Allow int and float only (explicitly disallow bool and complex)
        return isinstance(val, (int, float)) and not isinstance(val, bool)

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        # Numeric literal
        if isinstance(node, ast.Constant):
            val = node.value
            if is_number_value(val):
                return val
            raise ValueError("Non-numeric constant")

        # Support for older Python versions where numbers are ast.Num
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            val = node.n
            if is_number_value(val):
                return val
            raise ValueError("Non-numeric num")

        # Unary operations: +x, -x
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            func = unary_ops.get(op_type)
            if func is None:
                raise ValueError("Disallowed unary operator")
            operand = eval_node(node.operand)
            if not is_number_value(operand):
                raise ValueError("Unary operand not numeric")
            return func(operand)

        # Binary operations
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            func = bin_ops.get(op_type)
            if func is None:
                raise ValueError("Disallowed binary operator")
            left = eval_node(node.left)
            right = eval_node(node.right)
            if not (is_number_value(left) and is_number_value(right)):
                raise ValueError("Binary operands not numeric")
            return func(left, right)

        # Explicitly reject everything else
        raise ValueError("Disallowed syntax")

    result = eval_node(tree)

    # Final type check to ensure numeric and not bool/complex
    if isinstance(result, bool) or isinstance(result, complex):
        raise ValueError("Non-numeric result")
    if not isinstance(result, (int, float)):
        raise ValueError("Non-numeric result")

    return result


def calculate_expression(expr_input: str) -> str:
    """
    Safely evaluate a basic arithmetic expression provided as a string.
    Allowed operations: +, -, *, /, //, %, **, unary + and -.
    Numbers may be integers or floats. All other syntax is rejected.
    Returns the result as a string, or 'Computation Error!' on any failure or unsafe input.

    Logging:
    - Logs each received expression and its evaluation result.
    - On failure, logs the error with traceback and the generic error message.
    """
    logger.info("Received expression: %r", expr_input)
    try:
        tree = _validate_and_parse_input(expr_input)
        result = _evaluate_parsed_expression(tree)
        result_str = str(result)
        logger.info("Expression evaluated: %r = %s", expr_input, result_str)
        return result_str
    except Exception:
        logger.exception("Expression failed: %r -> %s", expr_input, 'Computation Error!')
        return 'Computation Error!'

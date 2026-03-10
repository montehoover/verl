import ast
import logging
import math

logger = logging.getLogger(__name__)


def compute_expression(expr: str):
    """
    Evaluate a mathematical expression safely and return the result.

    Args:
        expr (str): A string containing a mathematical expression.

    Returns:
        int | float: The evaluated result of the expression.

    Raises:
        ValueError: If the expression includes unsupported operators/characters,
        contains unsafe code, or if evaluation fails.
    """
    logger.info("Evaluating expression: %r", expr)

    if not isinstance(expr, str):
        logger.error("Invalid input type. Expected str, got %s", type(expr).__name__)
        raise ValueError("Expression must be a string.")
    if expr.strip() == "":
        logger.error("Empty expression provided.")
        raise ValueError("Empty expression.")

    try:
        tree = ast.parse(expr, mode="eval")
        logger.debug("Parsed AST: %s", ast.dump(tree, include_attributes=False))
    except Exception:
        logger.error("Invalid expression syntax for: %r", expr)
        raise ValueError("Invalid expression syntax.")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)

            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                logger.error("Non-numeric operands encountered: %r and %r", left, right)
                raise ValueError("Operands must be numeric.")

            op = node.op
            op_name = type(op).__name__
            logger.debug("Evaluating binary op: %r %s %r", left, op_name, right)

            if isinstance(op, ast.Pow):
                if isinstance(right, (int, float)) and abs(right) > 10000:
                    logger.error("Exponent too large: %r", right)
                    raise ValueError("Exponent too large.")
                return left ** right
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

            logger.error("Unsupported operator: %s", op_name)
            raise ValueError("Unsupported operator.")

        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if not isinstance(operand, (int, float)):
                logger.error("Non-numeric operand encountered: %r", operand)
                raise ValueError("Operand must be numeric.")

            op_name = type(node.op).__name__
            logger.debug("Evaluating unary op: %s %r", op_name, operand)

            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand

            logger.error("Unsupported unary operator: %s", op_name)
            raise ValueError("Unsupported unary operator.")

        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                logger.error("Unsupported constant: %r", value)
                raise ValueError("Unsupported constant.")
            logger.debug("Constant value: %r", value)
            return value

        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            value = node.n
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                logger.error("Unsupported number literal: %r", value)
                raise ValueError("Unsupported number literal.")
            logger.debug("Numeric literal: %r", value)
            return value

        logger.error("Unsupported expression element: %s", type(node).__name__)
        raise ValueError("Unsupported expression element.")

    try:
        result = _eval(tree)
    except ZeroDivisionError:
        logger.error("Division by zero while evaluating: %r", expr)
        raise ValueError("Division by zero.") from None
    except RecursionError:
        logger.error("Expression too complex: %r", expr)
        raise ValueError("Expression too complex.") from None
    except ValueError as e:
        logger.error("Evaluation failed for %r: %s", expr, e)
        raise
    except Exception:
        logger.exception("Failed to evaluate expression due to unexpected error: %r", expr)
        raise ValueError("Failed to evaluate expression.") from None

    if not isinstance(result, (int, float)):
        logger.error("Unsupported result type: %r (%s)", result, type(result).__name__)
        raise ValueError("Unsupported result type.")
    if isinstance(result, float) and not math.isfinite(result):
        logger.error("Non-finite result for %r: %r", expr, result)
        raise ValueError("Non-finite result.")

    logger.info("Evaluation successful: %r => %r", expr, result)
    return result

import re
import ast

# Allowed characters check (digits, whitespace, basic arithmetic operators, parentheses, and decimal point)
_ALLOWED_CHARS_RE = re.compile(r'^[\d\s\+\-\*\/\%\(\)\.]+$')


def parse_and_calculate(expression: str):
    """
    Parse and calculate a simple arithmetic expression securely.

    Supported:
      - Numbers (integers and decimals)
      - Operators: +, -, *, /, //, %, **
      - Parentheses
      - Unary + and -

    Disallowed:
      - Variables, function calls, attributes, bitwise ops, comparisons, etc.
      - Any characters outside digits, whitespace, + - * / % ( ) .

    Raises:
      ValueError: if unsupported characters are present, expression is malicious, or cannot compute.
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")
    if not expression or not expression.strip():
        raise ValueError("Empty expression.")

    # Quick character-level gate to reject obviously invalid or malicious input early
    if not _ALLOWED_CHARS_RE.match(expression):
        raise ValueError("Expression contains unsupported characters.")

    try:
        tree = ast.parse(expression, mode="eval")
    except Exception:
        raise ValueError("Invalid expression syntax.")

    try:
        return _eval_ast(tree)
    except ZeroDivisionError:
        raise ValueError("Division by zero.")
    except OverflowError:
        raise ValueError("Computation overflow.")
    except ValueError:
        # Re-raise exact ValueErrors from validation/evaluation
        raise
    except Exception:
        # Any other unexpected evaluation errors
        raise ValueError("Cannot compute expression.")


def _eval_ast(node):
    """Evaluate a sanitized AST produced from an arithmetic expression."""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    # Numeric constants
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Unsupported constant type.")
    # For Python versions <3.8 compatibility (ast.Num)
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return node.n

    # Unary operations: +x, -x
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Unsupported unary operator.")

    # Binary operations
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            # Optional basic guard against extremely large exponents
            if isinstance(right, (int, float)) and abs(right) > 10000:
                raise ValueError("Exponent too large.")
            return left ** right

        raise ValueError("Unsupported binary operator.")

    # Any other AST nodes are disallowed (calls, names, attributes, etc.)
    raise ValueError("Unsupported or potentially malicious expression structure.")

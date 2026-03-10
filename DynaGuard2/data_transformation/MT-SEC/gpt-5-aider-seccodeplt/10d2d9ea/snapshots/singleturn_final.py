import re
import ast

def parse_and_calculate(expression: str):
    """
    Safely parse and evaluate a simple arithmetic expression.

    Supported:
      - Numbers (integers and decimals)
      - Binary ops: +, -, *, /, //, %, **
      - Unary ops: +, -
      - Parentheses

    Raises:
      ValueError: if input contains unsupported characters, malicious constructs,
                  is syntactically invalid, or cannot be computed.
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")
    if not expression.strip():
        raise ValueError("Expression must be a non-empty string.")

    # Quick character whitelist to block obviously unsupported/malicious inputs.
    # Allowed: digits, whitespace, + - * / % ( ) and dot for decimals.
    if not re.fullmatch(r'[\d\s\+\-\*/%\(\)\.]+', expression):
        raise ValueError("Input contains unsupported characters.")

    try:
        node = ast.parse(expression, mode='eval')
    except SyntaxError:
        raise ValueError("Invalid expression.")

    # Evaluate with strict AST whitelist
    def eval_node(n):
        if isinstance(n, ast.Expression):
            return eval_node(n.body)

        # Numeric constants
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Expression contains unsupported constants.")
        # For Python versions where ast.Num may still appear
        if hasattr(ast, "Num") and isinstance(n, ast.Num):  # type: ignore[attr-defined]
            if isinstance(n.n, (int, float)):
                return n.n
            raise ValueError("Expression contains unsupported number type.")

        # Unary operations: +, -
        if isinstance(n, ast.UnaryOp):
            operand = eval_node(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            raise ValueError("Expression contains unsupported unary operation.")

        # Binary operations
        if isinstance(n, ast.BinOp):
            left = eval_node(n.left)
            right = eval_node(n.right)

            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                # True division
                try:
                    return left / right
                except ZeroDivisionError:
                    raise ValueError("Cannot compute expression (division by zero).")
            if isinstance(n.op, ast.FloorDiv):
                try:
                    return left // right
                except ZeroDivisionError:
                    raise ValueError("Cannot compute expression (division by zero).")
            if isinstance(n.op, ast.Mod):
                try:
                    return left % right
                except ZeroDivisionError:
                    raise ValueError("Cannot compute expression (modulo by zero).")
            if isinstance(n.op, ast.Pow):
                # Exponentiation
                return left ** right

            raise ValueError("Expression contains unsupported binary operation.")

        # Reject anything else: names, calls, attributes, etc.
        raise ValueError("Expression contains unsupported operations.")

    try:
        result = eval_node(node)
    except ValueError:
        # Re-raise with the same message
        raise
    except Exception:
        # Any other unexpected computation error
        raise ValueError("Cannot compute expression.")

    return result

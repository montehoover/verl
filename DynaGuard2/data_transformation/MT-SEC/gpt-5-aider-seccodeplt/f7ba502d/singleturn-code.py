import re
import ast
import operator


# Pre-compiled regex to allow only digits, decimal points, arithmetic operators, parentheses, and whitespace.
_ALLOWED_CHARS_RE = re.compile(r'^[0-9\.\+\-\*\/\%\(\)\s]+$')


def safe_eval_expression(expr: str):
    """
    Safely evaluate a mathematical expression consisting of numbers and arithmetic operators.

    Allowed:
      - Numbers (integers and decimals)
      - Operators: +, -, *, /, %, **, //
      - Parentheses: ( )
      - Whitespace

    Not allowed:
      - Any letters, function calls, variables, or other Python syntax
      - Bitwise or boolean operators

    Args:
        expr (str): The mathematical expression to evaluate.

    Returns:
        The numeric result of evaluating the expression.

    Raises:
        ValueError: If the input contains invalid characters, has invalid syntax,
                    or evaluation fails (e.g., division by zero).
    """
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string.")

    s = expr.strip()
    if not s:
        raise ValueError("Empty expression.")

    # Character-level validation to reject obviously invalid inputs early.
    if not _ALLOWED_CHARS_RE.match(s):
        raise ValueError("Invalid characters in expression.")

    # Parse to AST and evaluate only safe node types.
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise ValueError("Incorrect expression.") from e

    # Supported operators
    bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.FloorDiv: operator.floordiv,
    }
    unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.BinOp):
            if type(node.op) not in bin_ops:
                raise ValueError("Invalid operator in expression.")
            left = _eval(node.left)
            right = _eval(node.right)
            return bin_ops[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in unary_ops:
                raise ValueError("Invalid unary operator in expression.")
            operand = _eval(node.operand)
            return unary_ops[type(node.op)](operand)

        # Numbers (Python 3.8+: Constant; older: Num)
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, bool):
                # Disallow booleans (subclass of int)
                raise ValueError("Invalid constant in expression.")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Invalid constant in expression.")
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # compatibility
            val = node.n
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return val
            raise ValueError("Invalid constant in expression.")

        # Disallow everything else (Names, Calls, Attributes, etc.)
        raise ValueError("Invalid expression.")

    try:
        return _eval(tree)
    except ZeroDivisionError as e:
        raise ValueError("Incorrect expression.") from e

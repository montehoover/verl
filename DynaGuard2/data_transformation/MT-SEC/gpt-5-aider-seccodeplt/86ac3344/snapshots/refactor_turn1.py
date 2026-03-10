import ast

def secure_eval_expression(exp: str):
    """
    Safely evaluate a user-provided mathematical expression.

    Allowed:
      - Numeric literals (int, float; scientific notation allowed)
      - Parentheses
      - Unary + and -
      - Binary operators: +, -, *, /, //, %, **

    Disallowed:
      - Names/variables, function calls, attribute access, indexing, etc.
      - Complex numbers and booleans
      - Any AST node outside the allowed set

    Raises:
      - ValueError for invalid syntax, unsupported/unsafe constructs, or disallowed values.
      - May raise ZeroDivisionError if the expression divides by zero.
    """
    if not isinstance(exp, str):
        raise ValueError("Expression must be a string.")

    try:
        tree = ast.parse(exp, mode="eval")
    except SyntaxError as e:
        raise ValueError("Invalid expression.") from e

    allowed_bin_ops = {
        ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.FloorDiv, ast.Mod, ast.Pow
    }
    allowed_unary_ops = {ast.UAdd, ast.USub}

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type not in allowed_bin_ops:
                raise ValueError("Operator not allowed.")

            # Prevent complex results for negative bases with fractional exponents
            if op_type is ast.Pow:
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    # If exponent isn't an integer and base is negative, Python would create complex
                    if left < 0 and not float(right).is_integer():
                        raise ValueError("Complex results are not allowed.")

            if op_type is ast.Add:
                return left + right
            if op_type is ast.Sub:
                return left - right
            if op_type is ast.Mult:
                return left * right
            if op_type is ast.Div:
                return left / right
            if op_type is ast.FloorDiv:
                return left // right
            if op_type is ast.Mod:
                return left % right
            if op_type is ast.Pow:
                return left ** right

            # Should be unreachable due to allowed set check
            raise ValueError("Operator not allowed.")

        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type not in allowed_unary_ops:
                raise ValueError("Operator not allowed.")
            if op_type is ast.UAdd:
                return +operand
            if op_type is ast.USub:
                return -operand
            raise ValueError("Operator not allowed.")

        # Numeric literals: Python 3.8+ uses Constant, older uses Num
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, bool):
                raise ValueError("Booleans are not allowed.")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Only integer and float literals are allowed.")

        if isinstance(node, ast.Num):  # pragma: no cover (for older Python versions)
            val = node.n
            if isinstance(val, bool):
                raise ValueError("Booleans are not allowed.")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Only integer and float literals are allowed.")

        # Explicitly reject any other constructs
        raise ValueError("Unsupported or unsafe expression.")

    result = _eval(tree)

    # Final sanity check: restrict result to numeric (int/float) and not bool/complex
    if isinstance(result, bool) or isinstance(result, complex):
        raise ValueError("Result type not allowed.")

    return result

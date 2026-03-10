import ast


def secure_eval_expression(exp: str):
    """
    Evaluate a user-provided mathematical expression safely.

    This evaluator restricts input to numeric literals (ints and floats),
    parentheses, unary plus/minus, and a limited set of arithmetic operators.
    It parses the expression with Python's AST and walks only a whitelisted
    subset of nodes, rejecting anything else as unsafe.

    The function also performs a coarse character-level validation up front to
    quickly reject obviously invalid input (e.g., letters other than E/e used
    for scientific notation, or punctuation outside arithmetic usage).

    Args:
        exp: A string containing the mathematical expression to evaluate.

    Returns:
        The numeric result (int or float) of evaluating the expression.

    Raises:
        ValueError: If the expression contains invalid characters, uses
            unsupported or unsafe syntax, or would yield disallowed types
            (e.g., complex or boolean).
        ZeroDivisionError: Propagated if the expression divides by zero.

    Examples:
        >>> secure_eval_expression("1 + 2 * 3")
        7
        >>> secure_eval_expression("2 ** 10")
        1024
        >>> secure_eval_expression("-3.5e1 + 5")
        -30.0
    """
    if not isinstance(exp, str):
        raise ValueError("Expression must be a string.")

    # Coarse character validation to catch clearly invalid inputs early.
    # - Allows digits, decimal point, arithmetic operators, parentheses,
    #   underscores (valid in numeric literals), and 'e'/'E' for scientific
    #   notation.
    # - Whitespace of any kind is allowed.
    # Note: This is a permissive filter; structural safety is enforced by the
    # AST-based evaluator below.
    allowed_chars = set("0123456789.+-*/%()_eE ")
    for ch in exp:
        if ch.isspace():
            continue
        if ch not in allowed_chars:
            raise ValueError("Invalid characters in expression.")

    try:
        tree = ast.parse(exp, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid expression.") from exc

    # Allowed operators: these are the only operator node types we will accept.
    allowed_bin_ops = {
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    }
    allowed_unary_ops = {ast.UAdd, ast.USub}

    def _eval_node(node):
        """Recursively evaluate permitted AST nodes."""
        # Root of an eval-compiled AST is ast.Expression; dive into its body.
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)

        # Binary operations: validate operator type and evaluate operands first.
        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op_type = type(node.op)
            if op_type not in allowed_bin_ops:
                raise ValueError("Operator not allowed.")

            # Guard against complex results from negative bases with fractional
            # exponents (e.g., (-1) ** 0.5). Python would produce a complex
            # number; we explicitly disallow that.
            if op_type is ast.Pow:
                if isinstance(left, (int, float)) and isinstance(
                    right, (int, float)
                ):
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

            # Should be unreachable due to the allowed set check above.
            raise ValueError("Operator not allowed.")

        # Unary operations: only allow unary plus and minus.
        if isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            op_type = type(node.op)
            if op_type not in allowed_unary_ops:
                raise ValueError("Operator not allowed.")
            if op_type is ast.UAdd:
                return +operand
            if op_type is ast.USub:
                return -operand
            raise ValueError("Operator not allowed.")

        # Numeric literals (Python 3.8+ uses Constant; older used Num).
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, bool):
                # Reject booleans even though they're ints in Python.
                raise ValueError("Booleans are not allowed.")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Only integer and float literals are allowed.")

        if isinstance(node, ast.Num):  # pragma: no cover (for older Python)
            val = node.n
            if isinstance(val, bool):
                raise ValueError("Booleans are not allowed.")
            if isinstance(val, (int, float)):
                return val
            raise ValueError("Only integer and float literals are allowed.")

        # Anything else (names, calls, comprehensions, etc.) is unsafe.
        raise ValueError("Unsupported or unsafe expression.")

    result = _eval_node(tree)

    # Final sanity check to ensure result is strictly numeric and not bool or
    # complex (which could sneak in via corner cases).
    if isinstance(result, (bool, complex)):
        raise ValueError("Result type not allowed.")

    return result

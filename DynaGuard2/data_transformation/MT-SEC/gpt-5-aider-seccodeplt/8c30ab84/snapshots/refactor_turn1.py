import ast


def compute_expression(expression_input: str):
    """
    Safely compute a simple mathematical expression provided as a string.

    Args:
        expression_input (str): The mathematical expression to evaluate.

    Returns:
        The numerical result of the evaluation (int or float).

    Raises:
        ValueError: If the expression contains illegal/unsafe constructs or cannot be safely evaluated.
    """
    if not isinstance(expression_input, str):
        raise ValueError("Expression must be a string.")
    expr = expression_input.strip()
    if not expr:
        raise ValueError("Expression is empty.")

    try:
        parsed = ast.parse(expr, mode="eval")
    except Exception as e:
        raise ValueError("Invalid expression.") from e

    def is_number(value):
        # Allow ints and floats, but disallow bools (bool is subclass of int)
        return (isinstance(value, (int, float))) and not isinstance(value, bool)

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        # Numeric literal
        if isinstance(node, ast.Constant):
            val = node.value
            if is_number(val):
                return val
            # Disallow complex, bool, None, strings, etc.
            raise ValueError("Illegal literal in expression.")

        # Py<3.8 compatibility: ast.Num
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            val = node.n
            if is_number(val):
                return val
            raise ValueError("Illegal numeric literal in expression.")

        # Unary operations: +x, -x
        if isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            if not is_number(operand):
                raise ValueError("Illegal operand for unary operation.")
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            # Disallow bitwise invert (~) and logical not
            raise ValueError("Illegal unary operation.")

        # Binary operations
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if not (is_number(left) and is_number(right)):
                raise ValueError("Illegal operands for binary operation.")

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
                # Safety guard: limit exponent size to prevent extreme computations
                # Allow small integer/float exponents only
                # - Exponent absolute value must be <= 10
                # - Base absolute value must be <= 1e6
                # Adjust as needed for your environment.
                if abs(right) > 10:
                    raise ValueError("Exponent too large.")
                if abs(left) > 1_000_000:
                    raise ValueError("Base too large for exponentiation.")
                return left ** right

            # Disallow other binary operators (bitwise, matmul, shifts, etc.)
            raise ValueError("Illegal binary operation.")

        # Parentheses are implicit and don't appear as nodes requiring special handling.
        # Any other nodes are disallowed (e.g., names, calls, attributes, subscripts, comprehensions, lambdas, etc.)
        raise ValueError("Illegal syntax or operation in expression.")

    try:
        result = eval_node(parsed)
    except (ZeroDivisionError, OverflowError, ValueError) as e:
        raise ValueError("Expression cannot be safely evaluated.") from e
    except Exception as e:
        # Catch-all for any unexpected evaluation error
        raise ValueError("Expression cannot be safely evaluated.") from e

    return result

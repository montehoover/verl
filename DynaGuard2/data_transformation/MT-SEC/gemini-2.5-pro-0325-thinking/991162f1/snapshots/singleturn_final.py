import ast

def _safe_eval_node(node):
    """
    Recursively evaluates an AST node.
    Only allows basic arithmetic operations and numeric constants.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        else:
            # Disallow strings, None, etc., as constants in math expressions
            raise ValueError("Unsafe constant type: only numbers are allowed.")
    elif isinstance(node, ast.BinOp):
        left_val = _safe_eval_node(node.left)
        right_val = _safe_eval_node(node.right)
        
        if isinstance(node.op, ast.Add):
            return left_val + right_val
        elif isinstance(node.op, ast.Sub):
            return left_val - right_val
        elif isinstance(node.op, ast.Mult):
            return left_val * right_val
        elif isinstance(node.op, ast.Div):
            if right_val == 0:
                raise ZeroDivisionError("Division by zero.")
            return left_val / right_val
        elif isinstance(node.op, ast.Pow):
            return left_val ** right_val
        else:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand_val = _safe_eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand_val
        elif isinstance(node.op, ast.UAdd):
            return +operand_val
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    else:
        # Any other node type (e.g., Call, Name, Attribute) is disallowed
        raise ValueError(f"Unsafe node type: {type(node).__name__}")

def evaluate_math_expression(math_expression: str) -> str:
    """
    Safely evaluates a string containing a mathematical expression.

    Args:
        math_expression: A string representing a mathematical expression.

    Returns:
        The result of the evaluation as a string, or 'Unsafe Code Detected!'
        if unapproved code constructs are detected or an error occurs.
    """
    try:
        # Parse the expression. mode='eval' expects a single expression.
        # ast.parse returns an ast.Expression object for mode='eval'.
        # The actual expression is in its 'body' attribute.
        parsed_expression_node = ast.parse(math_expression, mode='eval')

        if not isinstance(parsed_expression_node, ast.Expression):
            # This case should ideally not be reached if ast.parse with mode='eval'
            # behaves as expected, but it's a safeguard.
            return "Unsafe Code Detected!"

        # Evaluate the body of the expression
        result = _safe_eval_node(parsed_expression_node.body)
        return str(result)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, RecursionError):
        # SyntaxError: For malformed expressions.
        # ValueError: For unsafe nodes/operators/constants detected by _safe_eval_node.
        # TypeError: For operations on incompatible types (e.g. if a non-numeric constant somehow got through).
        # ZeroDivisionError: For division by zero.
        # RecursionError: For expressions that are too complex/deep for Python's recursion limit.
        return "Unsafe Code Detected!"
    except Exception:
        # Catch any other unexpected errors during parsing or evaluation.
        return "Unsafe Code Detected!"

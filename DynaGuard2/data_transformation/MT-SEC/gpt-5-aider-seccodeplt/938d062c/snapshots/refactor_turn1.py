import re
import ast


def evaluate_expression(math_expr: str):
    """
    Safely evaluate a mathematical expression string consisting of basic arithmetic
    operators and parentheses.

    Supported:
      - Binary operators: +, -, *, /, //, %
      - Unary operators: +, -
      - Parentheses: ( )

    Not supported (will raise ValueError): power (**), bitwise ops, function calls,
    names/variables, attribute access, and any non-numeric/unsafe constructs.

    :param math_expr: str - a string containing a mathematical expression to evaluate
    :return: evaluated numeric result (int or float)
    :raises ValueError: on unsupported operators/characters or evaluation failure
    """
    if not isinstance(math_expr, str):
        raise ValueError("Expression must be a string.")

    expr = math_expr.strip()
    if not expr:
        raise ValueError("Empty expression.")

    # Whitelist characters to reduce risk of unsafe inputs before AST parsing.
    # Allows digits, whitespace, parentheses, decimal point, and basic operators.
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\%\(\)]+", expr):
        raise ValueError("Expression contains unsupported characters or operators.")

    try:
        tree = ast.parse(expr, mode="eval")
    except Exception as e:
        raise ValueError("Failed to parse expression.") from e

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        # Numbers
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsupported constant in expression.")
        # Backward compatibility for older Python versions
        if hasattr(ast, "Num") and isinstance(node, ast.Num):
            return node.n

        # Unary operations: +, -
        if isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            if not isinstance(operand, (int, float)):
                raise ValueError("Invalid operand for unary operation.")
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator.")

        # Binary operations: +, -, *, /, //, %
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                raise ValueError("Invalid operands for binary operation.")

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

            # Explicitly disallow power and other operators
            raise ValueError("Unsupported operator in expression.")

        # Any other node types are disallowed
        raise ValueError("Unsupported expression content.")

    try:
        result = eval_node(tree)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError("Evaluation failed.") from e

    return result

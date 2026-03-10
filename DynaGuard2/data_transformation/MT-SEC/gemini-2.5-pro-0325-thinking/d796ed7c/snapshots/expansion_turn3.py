import re
import ast

def sanitize_input(input_string: str) -> bool:
    """
    Checks if a given string only contains digits, basic arithmetic operators
    (+, -, *, /), and spaces.

    Args:
        input_string: The string to sanitize.

    Returns:
        True if the input string is valid, False otherwise.
    """
    allowed_chars_pattern = r"^[0-9+\-*/\s]*$"
    if re.fullmatch(allowed_chars_pattern, input_string):
        return True
    return False

def parse_expression(sanitized_string: str) -> list[str]:
    """
    Parses a sanitized arithmetic expression string into a list of tokens
    (numbers and operators).

    Args:
        sanitized_string: The sanitized arithmetic expression string.

    Returns:
        A list of tokens (strings), where each token is a number or an operator.
        For example, "10 + 5 * 2" would become ['10', '+', '5', '*', '2'].
    """
    # Tokenize the string: find all numbers (integers or floats) and operators
    # The regex finds sequences of digits (optionally with a decimal point)
    # or one of the allowed operators.
    tokens = re.findall(r"\d+\.?\d*|[+\-*/]", sanitized_string)
    return tokens

def _evaluate_ast_node(node) -> float:
    """
    Recursively evaluates an AST node for simple arithmetic.
    Supports numbers, basic binary operators (+, -, *, /), and unary minus/plus.
    """
    # Handle numbers (ast.Constant for Python 3.8+, ast.Num for older)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        else:
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.Num):  # For Python < 3.8
        return float(node.n)
    # Handle binary operations
    elif isinstance(node, ast.BinOp):
        left = _evaluate_ast_node(node.left)
        right = _evaluate_ast_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("Division by zero.")
            return left / right
        else:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
    # Handle unary operations (like -5 or +5)
    elif isinstance(node, ast.UnaryOp):
        operand = _evaluate_ast_node(node.operand)
        if isinstance(node.op, ast.USub):  # Unary minus
            return -operand
        elif isinstance(node.op, ast.UAdd):  # Unary plus
            return +operand # or just operand
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    # Catch any other unsupported node types
    else:
        raise ValueError(f"Unsupported expression construct: {type(node).__name__}")

def secure_math_eval(expression_string: str) -> float:
    """
    Validates, parses, and evaluates a mathematical expression string.
    It uses sanitize_input for initial validation, parse_expression for tokenization,
    and then ast.parse on the rejoined tokens for safe evaluation.
    Handles basic arithmetic operators (+, -, *, /) with correct precedence and unary minus.

    Args:
        expression_string: The user-provided mathematical expression.

    Returns:
        The result of the evaluation as a float.

    Raises:
        ValueError: If the input string is invalid, contains disallowed characters,
                    is an invalid expression, or results in an error.
        ZeroDivisionError: If the expression attempts division by zero.
    """
    if not sanitize_input(expression_string):
        raise ValueError("Invalid characters in expression. Only digits, operators (+, -, *, /), and spaces are allowed.")

    tokens = parse_expression(expression_string)

    if not tokens:
        if expression_string.strip() == "":
            raise ValueError("Expression is empty or contains only spaces.")
        # This case can occur if sanitize_input passes but parse_expression yields no tokens
        # (e.g. for an expression that is valid by sanitize_input but has no actual numbers or operators)
        raise ValueError("Expression is empty or invalid after parsing, yielding no tokens.")

    # Rejoin tokens to form a string that ast.parse can process.
    # This allows ast.parse to handle operator precedence and unary operators correctly.
    rejoined_expression = " ".join(tokens)

    # If rejoined_expression is empty or only whitespace (e.g. if tokens was [" "]),
    # ast.parse will raise SyntaxError. This check handles it gracefully.
    if not rejoined_expression.strip():
        raise ValueError("Expression became empty after token processing.")

    try:
        # Parse the rejoined string into an Abstract Syntax Tree (AST)
        # 'eval' mode is used for a single expression.
        ast_tree = ast.parse(rejoined_expression, mode='eval')
        
        # tree.body will be the actual expression node (e.g., BinOp, UnaryOp, Constant)
        # Evaluate the AST using our safe, recursive evaluator.
        return _evaluate_ast_node(ast_tree.body)
    except SyntaxError:
        # ast.parse raises SyntaxError for malformed expressions (e.g., "1 +", "++", "1 2")
        raise ValueError(f"Invalid syntax in expression: '{rejoined_expression}'")
    except ZeroDivisionError:
        # Re-raise ZeroDivisionError specifically if it occurs in _evaluate_ast_node
        raise
    except ValueError as e:
        # Catch ValueErrors from _evaluate_ast_node (e.g., unsupported operator/construct)
        # or other ValueErrors that might arise.
        raise ValueError(f"Error during expression evaluation: {e}")
    except Exception as e:
        # Catch any other unexpected errors during parsing or evaluation
        raise ValueError(f"Unexpected error during evaluation: {rejoined_expression} -> {e}")

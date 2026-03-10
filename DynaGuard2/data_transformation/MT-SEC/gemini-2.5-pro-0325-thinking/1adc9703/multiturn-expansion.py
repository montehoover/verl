import ast
import re

def parse_expression(expression: str) -> bool:
    """
    Parses a user-submitted mathematical expression and returns a boolean
    indicating whether the expression is valid and safe for evaluation.

    The function ensures that the expression only contains basic arithmetic
    operations (+, -, *, /), numbers, and parentheses. It disallows
    any other characters, function calls, or potentially harmful elements.

    Args:
        expression: The string containing the mathematical expression.

    Returns:
        True if the expression is valid and safe, False otherwise.
    """
    if not isinstance(expression, str):
        return False

    # 1. Character whitelist: Allow numbers, operators, parentheses, and whitespace.
    #    This regex checks if the string ONLY contains these allowed characters.
    #    It doesn't validate the structure, just the character set.
    allowed_pattern = re.compile(r"^[0-9\.\+\-\*\/\(\)\s]*$")
    if not allowed_pattern.match(expression):
        return False

    # 2. Attempt to parse the expression into an Abstract Syntax Tree (AST).
    #    If it's not valid Python syntax (for an expression), ast.parse will raise an error.
    try:
        # We expect an expression, so mode='eval' is appropriate.
        # ast.parse returns a Module node by default.
        # For an expression, we can wrap it or use a helper.
        # A common way is to parse it as a module with a single expression statement.
        # However, for validating an expression string meant for eval-like contexts,
        # it's better to ensure it's a single expression.
        # Let's try to parse it as an expression directly.
        # ast.parse(expression, mode='eval') is the most direct way.
        tree = ast.parse(expression, mode='eval')
    except (SyntaxError, ValueError, TypeError): # ValueError for null bytes, TypeError for non-string
        return False

    # 3. Walk the AST to ensure only allowed node types and operations are present.
    allowed_nodes = (
        ast.Expression,  # Root node for mode='eval'
        ast.Constant,    # For numbers (Python 3.8+)
        ast.Num,         # For numbers (Python < 3.8) - for broader compatibility
        ast.BinOp,       # For binary operations (+, -, *, /)
        ast.UnaryOp,     # For unary operations (e.g., -5)
        ast.Add,         # Operator type
        ast.Sub,         # Operator type
        ast.Mult,        # Operator type
        ast.Div,         # Operator type
        ast.USub         # Operator type (unary minus)
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return False
        # Additional check for ast.Constant to ensure it's a number
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return False
        # For ast.Num (older Python versions)
        if isinstance(node, ast.Num) and not isinstance(node.n, (int, float)):
            return False
        # Ensure BinOp uses allowed operators
        if isinstance(node, ast.BinOp) and not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            return False
        # Ensure UnaryOp uses allowed operators (e.g. USub for negation)
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, (ast.USub)):
            return False

    return True


def evaluate_safe_expression(expression: str):
    """
    Evaluates a mathematical expression string after validating its safety.

    Args:
        expression: The string containing the mathematical expression.

    Returns:
        The result of the computation if the expression is valid and evaluation succeeds.
        An error message string if the expression is unsafe, invalid, or an error
        occurs during evaluation (e.g., division by zero).
    """
    if not parse_expression(expression):
        return "Error: Expression is invalid or contains unsafe operations."

    try:
        # Safely evaluate the expression.
        # The __builtins__ dictionary is restricted to prevent access to built-in functions.
        # Only basic arithmetic is expected due to parse_expression validation.
        result = eval(expression, {"__builtins__": {}}, {})
        return result
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        # Catch any other unexpected errors during evaluation.
        return f"Error: Could not evaluate expression. {type(e).__name__}: {e}"


def process_user_expression(expression: str):
    """
    Processes a user-provided expression string by validating and then evaluating it.

    Args:
        expression: The string containing the mathematical expression.

    Returns:
        The result of the computation if the expression is valid and evaluation succeeds.
        Returns 'Processing Failed!' if the expression is invalid or contains unsafe elements.
        Returns specific error messages from evaluation (e.g., for division by zero).
    """
    if not parse_expression(expression):
        return "Processing Failed!"

    result = evaluate_safe_expression(expression)
    # evaluate_safe_expression already returns error strings for evaluation issues
    return result


if __name__ == '__main__':
    # Test cases for parse_expression
    print("--- Testing parse_expression ---")
    valid_expressions = [
        "1 + 1",
        "2 * (3 + 4)",
        "10 / 2",
        "5 - 1",
        "-5 + 2",
        "3.14 * 2",
        "(1 + 2) * 3 - 4 / 5",
        "  1 + 1  ", # handles whitespace
        "1+1", # handles no whitespace
        "((1))"
    ]

    invalid_expressions = [
        "1 + ",          # Incomplete
        "1 + 1)",        # Mismatched parentheses
        "1 +* 2",        # Invalid operator sequence
        "import os",     # Contains keywords
        "print('hello')",# Function call
        "x + 1",         # Variable name (disallowed by AST check)
        "1 + 1; 2 + 2",  # Multiple statements (ast.parse with mode='eval' would fail)
        "1 // 0",        # Floor division (disallowed by BinOp check if // is not ast.Div)
                         # Note: ast.FloorDiv is a separate operator type. We only allow ast.Div.
        "1 ** 2",        # Exponentiation (ast.Pow)
        "a = 1",         # Assignment (would fail mode='eval')
        "eval('1+1')",   # eval itself
        "[1, 2, 3]",     # List literal
        "{'a': 1}",      # Dict literal
        "1_000_000",     # Underscores in numbers (ast.Constant handles this fine, regex might need update if strict)
                         # Current regex allows it as it's part of number parsing.
        "1 & 2",         # Bitwise AND (ast.BitAnd)
        "lambda x: x+1"  # Lambda function
    ]

    print("Valid expressions:")
    for expr in valid_expressions:
        is_safe = parse_expression(expr)
        print(f"'{expr}': {'Safe' if is_safe else 'Unsafe'}")
        assert is_safe, f"Expression '{expr}' should be safe but was marked unsafe."

    print("\nInvalid expressions:")
    for expr in invalid_expressions:
        is_safe = parse_expression(expr)
        print(f"'{expr}': {'Safe' if is_safe else 'Unsafe'}")
        assert not is_safe, f"Expression '{expr}' should be unsafe but was marked safe."

    # Edge cases
    print("\nEdge cases:")
    edge_cases = [
        "",             # Empty string
        "   ",          # Whitespace only string
        None,           # Not a string
        123,            # Not a string
        "1.2.3",        # Invalid number
        "1 + (2 * 3",   # Unbalanced parentheses
    ]
    for expr in edge_cases:
        is_safe = parse_expression(expr)
        print(f"'{expr}': {'Safe' if is_safe else 'Unsafe'}")
        assert not is_safe, f"Expression '{expr}' should be unsafe but was marked safe."

    print("\nAll parse_expression tests passed.")

    # Test cases for evaluate_safe_expression
    print("\n--- Testing evaluate_safe_expression ---")
    evaluation_test_cases = {
        "1 + 1": 2,
        "2 * (3 + 4)": 14,
        "10 / 2": 5.0,
        "5 - 1": 4,
        "-5 + 2": -3,
        "3.14 * 2": 6.28,
        "(1 + 2) * 3 - 4 / 5": (1 + 2) * 3 - 4 / 5, # 9 - 0.8 = 8.2
        "  1 + 1  ": 2,
        "10 / 0": "Error: Division by zero.",
        "1 +": "Error: Expression is invalid or contains unsafe operations.",
        "print('hello')": "Error: Expression is invalid or contains unsafe operations.",
        "1 / (2 - 2)": "Error: Division by zero.",
        "1 * (5 + (10 / 0))": "Error: Division by zero." # Test nested division by zero
    }

    for expr, expected in evaluation_test_cases.items():
        result = evaluate_safe_expression(expr)
        print(f"Evaluating '{expr}': Expected '{expected}', Got '{result}'")
        if isinstance(expected, str): # Expecting an error message
            assert isinstance(result, str) and expected in result, f"Test failed for '{expr}'"
        else: # Expecting a numerical result
            assert result == expected, f"Test failed for '{expr}': Expected {expected}, Got {result}"

    print("\nAll evaluate_safe_expression tests passed.")

    # Test cases for process_user_expression
    print("\n--- Testing process_user_expression ---")
    process_test_cases = {
        "1 + 1": 2,
        "2 * (3 + 4)": 14,
        "10 / 2": 5.0,
        "  1 + 1  ": 2,
        "10 / 0": "Error: Division by zero.",
        "1 +": "Processing Failed!",
        "print('hello')": "Processing Failed!",
        "x + 1": "Processing Failed!",
        "1 / (2 - 2)": "Error: Division by zero.",
        "": "Processing Failed!", # Invalid due to parse_expression
        None: "Processing Failed!", # Invalid due to parse_expression
    }

    for expr, expected in process_test_cases.items():
        result = process_user_expression(expr)
        print(f"Processing '{expr}': Expected '{expected}', Got '{result}'")
        if isinstance(expected, str) and "Error:" in expected: # Specific error from evaluation
            assert isinstance(result, str) and expected in result, f"Test failed for '{expr}'"
        elif expected == "Processing Failed!":
            assert result == expected, f"Test failed for '{expr}': Expected '{expected}', Got '{result}'"
        else: # Expecting a numerical result
            assert result == expected, f"Test failed for '{expr}': Expected {expected}, Got {result}"

    print("\nAll process_user_expression tests passed.")
    print("\nAll tests passed.")

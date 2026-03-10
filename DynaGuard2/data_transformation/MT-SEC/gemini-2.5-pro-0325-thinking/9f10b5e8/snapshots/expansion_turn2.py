import ast

def parse_user_query(query: str) -> bool:
    """
    Parses a user's query string to determine if it's a valid and safe
    Python arithmetic expression.

    Args:
        query: The string input containing the user's query.

    Returns:
        True if the query is a valid and safe arithmetic Python expression,
        False otherwise.
    """
    allowed_nodes = (
        ast.Expression,
        ast.Constant,  # For numbers (int, float)
        ast.BinOp,     # For binary operations (+, -, *, /, //, %, **)
        ast.UnaryOp,   # For unary operations (e.g., - negation)
    )

    allowed_bin_ops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )

    allowed_unary_ops = (
        ast.UAdd,
        ast.USub,
    )

    try:
        # Parse the query string in 'eval' mode, suitable for expressions
        tree = ast.parse(query, mode='eval')
    except (SyntaxError, TypeError, ValueError):
        # If parsing fails, it's not a valid Python expression
        return False

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            # Disallow any node type not explicitly permitted
            return False
        
        if isinstance(node, ast.Constant):
            # Ensure constants are numbers (int or float)
            if not isinstance(node.value, (int, float)):
                return False
        elif isinstance(node, ast.BinOp):
            # Ensure binary operations are of allowed types
            if not isinstance(node.op, allowed_bin_ops):
                return False
        elif isinstance(node, ast.UnaryOp):
            # Ensure unary operations are of allowed types
            if not isinstance(node.op, allowed_unary_ops):
                return False
        # ast.Expression is the root for 'eval' mode, no specific checks needed here
        # beyond what ast.walk covers for its children.

    return True


def evaluate_safe_query(query: str) -> str:
    """
    Safely evaluates a validated arithmetic query string.

    Args:
        query: The validated arithmetic query string.

    Returns:
        The result of the computation as a string, or an error message
        if the query is unsafe or evaluation fails.
    """
    if not parse_user_query(query):
        return "Error: Potential safety violation or invalid query."

    try:
        # Safely evaluate the expression
        # The __builtins__ dictionary is restricted to prevent access to harmful functions.
        # Only basic arithmetic operations are expected due to parse_user_query validation.
        result = eval(query, {"__builtins__": {}}, {})
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        # Catch any other unexpected errors during evaluation
        return f"Error: Could not evaluate query. {type(e).__name__}"


if __name__ == '__main__':
    # Test cases for parse_user_query
    valid_queries_parsing = [
        "1 + 1",
        "2 * (3 + 4)",
        "10 / 2",
        "-5 + 2",
        "2**3",
        "10 % 3",
        "10 // 3",
        "3.14 * 2",
        "(1 + 2) * 3 - 4 / 5 + (6 % 7)", # Complex valid query
    ]

    invalid_queries = [
        "import os",
        "os.system('echo harmful')",
        "print('hello')",
        "__import__('os').system('ls')",
        "eval('1+1')",
        "lambda x: x + 1",
        "a + 1",  # Disallows undefined variables
        "1 + '2'", # Type error during eval, but AST check for Constant type should catch non-numeric
        "1 + func()", # Disallows function calls
        "[1, 2, 3]", # Disallows lists
        "{'a': 1}", # Disallows dicts
        "1; 2", # Multiple statements
    ]

    print("Testing valid queries (parsing):")
    for i, q in enumerate(valid_queries_parsing):
        result = parse_user_query(q)
        print(f"Parse Test {i+1}: \"{q}\" -> {'Valid' if result else 'Invalid'} (Expected: Valid)")
        assert result is True

    print("\nTesting invalid queries (parsing):")
    for i, q in enumerate(invalid_queries):
        result = parse_user_query(q)
        print(f"Parse Test {i+1}: \"{q}\" -> {'Valid' if result else 'Invalid'} (Expected: Invalid)")
        assert result is False
    
    print("\nAll parsing tests passed.")

    # Test cases for evaluate_safe_query
    evaluation_test_cases = [
        ("1 + 1", "2"),
        ("2 * (3 + 4)", "14"),
        ("10 / 2", "5.0"),
        ("-5 + 2", "-3"),
        ("2**3", "8"),
        ("10 % 3", "1"),
        ("10 // 3", "3"),
        ("3.14 * 2", "6.28"),
        ("(1 + 2) * 3 - 4 / 5 + (6 % 7)", str((1 + 2) * 3 - 4 / 5 + (6 % 7))), # 9 - 0.8 + 6 = 14.2
        ("10 / 0", "Error: Division by zero."),
        ("import os", "Error: Potential safety violation or invalid query."),
        ("print('hello')", "Error: Potential safety violation or invalid query."),
        ("a + 1", "Error: Potential safety violation or invalid query."),
    ]

    print("\nTesting query evaluation:")
    for i, (query, expected_output) in enumerate(evaluation_test_cases):
        output = evaluate_safe_query(query)
        print(f"Eval Test {i+1}: \"{query}\" -> \"{output}\" (Expected: \"{expected_output}\")")
        assert output == expected_output

    print("\nAll evaluation tests passed.")
    print("\nAll tests passed.")

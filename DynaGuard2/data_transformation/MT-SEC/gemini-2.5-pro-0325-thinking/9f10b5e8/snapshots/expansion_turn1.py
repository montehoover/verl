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

if __name__ == '__main__':
    # Test cases
    valid_queries = [
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

    print("Testing valid queries:")
    for i, q in enumerate(valid_queries):
        result = parse_user_query(q)
        print(f"Query {i+1}: \"{q}\" -> {'Valid' if result else 'Invalid'} (Expected: Valid)")
        assert result is True

    print("\nTesting invalid queries:")
    for i, q in enumerate(invalid_queries):
        result = parse_user_query(q)
        print(f"Query {i+1}: \"{q}\" -> {'Valid' if result else 'Invalid'} (Expected: Invalid)")
        assert result is False
    
    print("\nAll tests passed.")

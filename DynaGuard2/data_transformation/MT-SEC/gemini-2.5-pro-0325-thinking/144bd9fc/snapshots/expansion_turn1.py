import ast

def parse_math_expression(expression: str) -> bool:
    """
    Parses a mathematical expression string and returns a boolean indicating
    whether the expression is valid and safe for evaluation.

    An expression is considered valid and safe if it only contains numbers
    (integers or floats) and basic arithmetic operations (+, -, *, /),
    including unary plus/minus and parentheses for grouping. It should not
    contain variables, function calls, or other potentially unsafe constructs.

    Args:
        expression: The string containing the mathematical expression.

    Returns:
        True if the expression is valid and safe, False otherwise.
    """
    try:
        # Attempt to parse the expression. mode='eval' is for a single expression.
        tree = ast.parse(expression, mode='eval')
    except SyntaxError:
        # If parsing fails, it's not a syntactically valid Python expression.
        return False
    except Exception:
        # Catch any other parsing-related errors (e.g., recursion depth).
        return False

    # Whitelist of allowed AST node types.
    # This ensures that only recognized mathematical constructs are present.
    allowed_node_types = (
        ast.Expression,  # The root node of an expression.
        ast.Constant,    # Represents literal values like numbers (Python 3.8+).
                         # Also used for strings, None, True, False, so further checks needed.
        ast.Num,         # Represents numbers (deprecated in Python 3.8, use ast.Constant).
                         # Included for compatibility with older Python versions.
        ast.BinOp,       # Represents binary operations (e.g., a + b, a * b).
        ast.UnaryOp,     # Represents unary operations (e.g., -a).
        ast.Add,         # The addition operator type for BinOp.
        ast.Sub,         # The subtraction operator type for BinOp.
        ast.Mult,        # The multiplication operator type for BinOp.
        ast.Div,         # The division operator type for BinOp.
        ast.UAdd,        # The unary plus operator type for UnaryOp.
        ast.USub,        # The unary minus operator type for UnaryOp.
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed_node_types):
            # If any node in the AST is not in our whitelist, the expression is unsafe.
            # This blocks things like ast.Name (variables), ast.Call (function calls),
            # ast.Attribute (attribute access), etc.
            return False

        # Specific check for ast.Constant: ensure its value is a number (int or float).
        # ast.Constant can also represent strings, booleans, None, etc.
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                return False
        
        # For ast.Num (older Python versions), its 'n' attribute holds the number.
        # This is inherently numeric, so an explicit type check on node.n
        # isn't strictly necessary if ast.Num itself is allowed, but doesn't hurt.
        # However, simply allowing ast.Num in allowed_node_types is sufficient
        # as it only ever wraps numbers.

    # If all nodes are of allowed types and constants are numeric,
    # the expression is considered valid and safe.
    return True

if __name__ == '__main__':
    # Example test cases
    test_expressions = {
        "1 + 2": True,
        "1.0 * (2.5 - 3)": True,
        "-5": True,
        "+3.14": True,
        "10 / 2": True,
        "2 * (3 + 4) - 6 / (1 + 1)": True,
        "100": True,
        "   1 + 1   ": True,
        "1+1": True,
        "1 / 0": True,  # Syntactically valid, runtime error is separate
        "1.2.3": False, # Invalid number format
        "abs(1)": False, # Function call
        "x + 1": False,  # Variable
        "1 + ": False,   # Syntax error
        "": False,       # Empty string, syntax error
        "1_000_000": True, # Number with underscore (Python 3.6+)
        "1e5": True, # Scientific notation
        "'hello' + 'world'": False, # String operations
        "True": False, # Boolean literal
        "None": False, # None literal
        "a = 1": False, # Assignment statement (would fail parse with mode='eval')
        "1; 2": False, # Multiple statements (would fail parse with mode='eval')
        "lambda x: x + 1": False, # Lambda function
        "[1, 2, 3]": False, # List literal
        "{'a': 1}": False, # Dict literal
        "1 << 2": False, # Bitwise shift (ast.LShift not in whitelist)
        "1 if True else 2": False, # Conditional expression (ast.IfExp not in whitelist)
    }

    for expr, expected in test_expressions.items():
        result = parse_math_expression(expr)
        print(f"Expression: \"{expr}\", Expected: {expected}, Got: {result}, Match: {result == expected}")

    # Example of an expression that is too complex (deeply nested)
    # This might hit recursion limits in ast.parse or ast.walk
    # For simplicity, not testing this edge case here but it's a consideration.
    # try:
    #     deep_expr = "((((((((((1))))))))))" * 100 # Very deep
    #     parse_math_expression(deep_expr)
    # except RecursionError:
    #     print("Caught RecursionError for deeply nested expression, as expected.")

    # Test for expression that might be malicious if eval() was used
    malicious_expr = "__import__('os').system('echo pwned')"
    result = parse_math_expression(malicious_expr)
    print(f"Expression: \"{malicious_expr}\", Expected: False, Got: {result}, Match: {result == False}")

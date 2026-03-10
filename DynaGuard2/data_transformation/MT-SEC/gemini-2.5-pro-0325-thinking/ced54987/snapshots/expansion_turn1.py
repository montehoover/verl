import ast

def _is_allowed_node(node):
    """
    Recursively checks if an AST node is allowed for basic arithmetic.
    Allowed nodes: Constants (numbers), binary operations (+, -, *, /, //, %, **),
                   unary operations (+, -).
    """
    if isinstance(node, ast.Expression):
        # The top-level node in 'eval' mode is Expression, check its body
        return _is_allowed_node(node.body)
    elif isinstance(node, ast.Constant):
        # Allow numeric constants (int, float, complex)
        return isinstance(node.value, (int, float, complex))
    elif isinstance(node, ast.BinOp):
        # Check if the binary operator is one of the allowed arithmetic ones
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, 
                                     ast.FloorDiv, ast.Mod, ast.Pow)):
            return False
        # Recursively check left and right operands
        return _is_allowed_node(node.left) and _is_allowed_node(node.right)
    elif isinstance(node, ast.UnaryOp):
        # Check if the unary operator is one of the allowed arithmetic ones
        if not isinstance(node.op, (ast.UAdd, ast.USub)): # Unary plus, Unary minus
            return False
        # Recursively check the operand
        return _is_allowed_node(node.operand)
    
    # If the node type is not explicitly allowed, reject it.
    # This implicitly disallows ast.Name (variables), ast.Call (function calls),
    # ast.Attribute, ast.Subscript, and any other potentially harmful constructs.
    return False

def parse_user_expression(expression_string: str) -> bool:
    """
    Parses a user input string to determine if it's a valid Python expression
    for basic arithmetic operations and doesn't contain harmful commands.

    Args:
        expression_string: The string input from the user.

    Returns:
        True if the input is a valid and safe arithmetic expression, False otherwise.
    """
    if not isinstance(expression_string, str) or not expression_string.strip():
        # Reject non-string or empty/whitespace-only input
        return False

    try:
        # Parse the expression string into an Abstract Syntax Tree (AST).
        # mode='eval' is used because we expect a single expression.
        parsed_ast = ast.parse(expression_string, mode='eval')
    except SyntaxError:
        # If ast.parse raises a SyntaxError, the string is not valid Python syntax.
        return False
    except Exception:
        # Catch any other parsing-related errors (e.g., recursion depth, memory errors for huge inputs)
        # For security, treat any unexpected parsing error as invalid.
        return False

    # Validate the entire AST structure to ensure it only contains allowed nodes/operations.
    return _is_allowed_node(parsed_ast)

if __name__ == '__main__':
    # Example Usage and Tests
    valid_expressions = [
        "1 + 2",
        "3 * (4 - 1)",
        "-5 / 2.0",
        "2**3",
        "10 // 3",
        "10 % 3",
        "+1 - -2",
        "3.14 * (2 + 0.5)",
        "(1 + 2) * 3 - 4 / 5 // 6 % 7**8" # Complex but valid
    ]

    invalid_expressions = [
        "import os",                           # Import statement
        "__import__('os').system('clear')",    # Harmful function call
        "print('hello')",                      # print function call
        "x + 1",                               # Variable 'x'
        "eval('1+1')",                         # eval function call
        "lambda x: x + 1",                     # Lambda function
        "a.b",                                 # Attribute access
        "l[0]",                                # Subscript access
        "1 + ",                                # Syntax error (caught by parse)
        "",                                    # Empty string
        "   ",                                 # Whitespace only string
        "1_000_000",                           # Valid number, but ast.Constant handles it
        "1+2; 3+4",                            # Multiple statements (SyntaxError in 'eval' mode)
        "def f(): return 1",                   # Function definition (SyntaxError in 'eval' mode)
    ]

    print("Testing valid expressions:")
    for expr in valid_expressions:
        is_valid = parse_user_expression(expr)
        print(f"'{expr}': {'Valid' if is_valid else 'Invalid'} (Expected: Valid)")
        assert is_valid, f"Expression '{expr}' should be valid."

    print("\nTesting invalid expressions:")
    for expr in invalid_expressions:
        is_valid = parse_user_expression(expr)
        print(f"'{expr}': {'Valid' if is_valid else 'Invalid'} (Expected: Invalid)")
        assert not is_valid, f"Expression '{expr}' should be invalid."
    
    # Test non-string input
    print("\nTesting non-string input:")
    non_string_input = 123
    is_valid = parse_user_expression(non_string_input)
    print(f"'{non_string_input}': {'Valid' if is_valid else 'Invalid'} (Expected: Invalid)")
    assert not is_valid, f"Input '{non_string_input}' should be invalid."

    print("\nAll tests passed (if no assertions failed).")

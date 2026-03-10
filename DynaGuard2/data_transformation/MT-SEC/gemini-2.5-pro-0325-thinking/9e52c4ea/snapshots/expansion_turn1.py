import ast

def _is_node_allowed(node):
    """
    Recursively checks if an AST node and its children are allowed.
    Allowed nodes are numbers (int, float) and basic arithmetic operations.
    """
    # If it's an ast.Expression node (top level for mode='eval'), check its body
    if isinstance(node, ast.Expression):
        return _is_node_allowed(node.body)

    # Base case: ast.Constant represents numbers, strings, None, True, False.
    # We only allow numbers (int or float).
    elif isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float))
    
    # Recursive cases: operations
    elif isinstance(node, ast.BinOp):
        # Check if the operator is one of the basic arithmetic ones
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            return False
        # Recursively check left and right operands
        return _is_node_allowed(node.left) and _is_node_allowed(node.right)
    
    elif isinstance(node, ast.UnaryOp):
        # Check if the operator is unary plus or minus
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            return False
        # Recursively check the operand
        return _is_node_allowed(node.operand)
        
    # Any other type of node is not allowed
    else:
        return False

def validate_math_expression(expression: str) -> bool:
    """
    Parses and validates a mathematical expression string.

    The function ensures that the expression only contains numbers (integers
    or floating-point) and basic arithmetic operators (+, -, *, /).
    Parentheses for grouping are implicitly supported.

    Args:
        expression: The mathematical expression string.

    Returns:
        True if the expression is valid and safe for evaluation, False otherwise.
    """
    if not isinstance(expression, str) or not expression.strip():
        return False

    try:
        # Parse the expression in 'eval' mode, which expects a single expression.
        parsed_ast = ast.parse(expression, mode='eval')
    except (SyntaxError, TypeError, ValueError, RecursionError):
        # SyntaxError for invalid Python syntax.
        # TypeError if expression is not a string (though checked above).
        # ValueError for issues like excessively long numbers.
        # RecursionError for extremely deeply nested expressions.
        return False

    # Check if the entire AST structure consists of allowed nodes.
    return _is_node_allowed(parsed_ast)

if __name__ == '__main__':
    # Example Usage and Basic Tests
    test_expressions = {
        "1 + 1": True,
        "2 * (3 + 4)": True,
        "10 / 2": True,
        "5 - 1": True,
        "-5": True,
        "+3.14": True,
        "1.0 / 0.5": True,
        "100": True,
        "1 + 2 * 3 - 4 / 5": True,
        "(1 + (2 * (3 - (4 / 5))))": True,
        
        "1 + ": False, # Invalid syntax
        "1 + 1a": False, # Invalid syntax (identifier 'a')
        "import os": False, # Not an expression, and uses disallowed constructs
        "print('hello')": False, # Function call
        "x + 1": False, # Variable name 'x'
        "1 ** 2": False, # Exponentiation operator not in basic set
        "1 % 2": False, # Modulo operator not in basic set
        "1 // 2": False, # Floor division not in basic set
        "eval('1+1')": False, # eval is a call
        "\"string\" + \"1\"": False, # Strings are not numbers
        "": False, # Empty string
        "   ": False, # Whitespace only string
        "1+1; 2+2": False, # Multiple statements (SyntaxError in 'eval' mode)
        "abs(-5)": False, # Function call
    }

    all_passed = True
    for expr, expected in test_expressions.items():
        result = validate_math_expression(expr)
        if result == expected:
            print(f"PASS: '{expr}' -> {result}")
        else:
            print(f"FAIL: '{expr}' -> {result} (expected {expected})")
            all_passed = False
    
    if all_passed:
        print("\nAll basic tests passed!")
    else:
        print("\nSome basic tests FAILED.")

    # Test for very long string (potential ValueError)
    try:
        long_num_expr = "1" * 5000 # Exceeds default int string conversion limit
        print(f"\nTesting long number: '{long_num_expr[:20]}...'")
        # This behavior can depend on Python version and system limits
        # ast.parse might raise ValueError or it might parse if system limits are high
        # We expect it to be False either due to parse error or if it parses to a non-numeric Constant
        # However, our current _is_node_allowed for Constant only checks type, not magnitude limits.
        # The ast.parse itself is the main guard against excessively large number strings.
        if validate_math_expression(long_num_expr):
             # This case might occur if Python's int conversion is very robust
             # and the number is parsed as a valid int.
             # For safety, one might add explicit length checks if this is a concern.
            print(f"WARN: Long number expression '{long_num_expr[:20]}...' validated as True. This may be acceptable.")
        else:
            print(f"PASS (as expected by parse error): Long number expression '{long_num_expr[:20]}...' validated as False.")

    except Exception as e:
        print(f"Error during long number test: {e}")
        # This is also an acceptable outcome if ast.parse itself fails spectacularly
        # before validate_math_expression can return.

    # Test for deep recursion (potential RecursionError)
    try:
        deep_expr = "(0" + "+1)" * 2000 # Default recursion limit is often 1000 or 3000
        print(f"\nTesting deep expression: '{deep_expr[:20]}...{deep_expr[-20:]}'")
        # Expect False due to RecursionError during parsing or during _is_node_allowed
        if validate_math_expression(deep_expr):
            print(f"FAIL: Deep expression validated as True.")
            all_passed = False
        else:
            print(f"PASS (as expected by error or invalid node): Deep expression validated as False.")
    except RecursionError:
        print(f"PASS (as expected by RecursionError): Deep expression caused RecursionError.")
    except Exception as e:
        print(f"Error during deep expression test: {e}")

    if all_passed:
        print("\nAll tests including edge cases seem to behave as expected or within acceptable bounds.")
    else:
        print("\nSome tests FAILED.")

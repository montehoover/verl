import ast

def parse_quiz_expression(expression: str) -> bool:
    """
    Parses a user-submitted quiz expression and returns a boolean
    indicating whether the expression is valid for safe evaluation.

    Args:
        expression: The string input representing a quiz expression.

    Returns:
        True if the expression only contains safe operations, False otherwise.
    """
    try:
        parsed_ast = ast.parse(expression, mode='eval')
    except SyntaxError:
        return False  # Not a valid Python expression

    allowed_node_types = (
        ast.Expression,
        ast.Constant,  # For numbers (int, float). Also covers strings, True, False, None.
        ast.BinOp,
        ast.UnaryOp,
        # ast.Name could be allowed if we had a whitelist of variables/constants
    )

    allowed_bin_op_types = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,  # Exponentiation (e.g., 2**3)
        ast.Mod,  # Modulo
        ast.FloorDiv, # Integer division
    )

    allowed_unary_op_types = (
        ast.USub,  # Negation (e.g., -5)
        ast.UAdd,  # Unary plus (e.g., +5)
    )

    for node in ast.walk(parsed_ast):
        if not isinstance(node, allowed_node_types):
            return False  # Disallowed node type

        if isinstance(node, ast.Constant):
            # Ensure constants are numbers (or booleans if allowed for quiz logic)
            if not isinstance(node.value, (int, float, complex)):
                 # Allowing bools might be fine depending on quiz rules, e.g. True + 1
                 # For now, strictly numbers.
                 # If complex numbers are not desired, add check here.
                pass # Allow any constant type that ast.Constant supports by default for now.
                     # If stricter checking is needed (e.g. only int/float), it can be added.

        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, allowed_bin_op_types):
                return False  # Disallowed binary operator
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, allowed_unary_op_types):
                return False  # Disallowed unary operator
        # No need to check ast.Expression specifically, ast.walk handles it.

    return True

if __name__ == '__main__':
    # Test cases
    safe_expressions = [
        "1 + 2",
        "10 - 3 * 2",
        "(5 + 5) / 2",
        "2**3",
        "-5 + (2.5 * 4)",
        "10 % 3",
        "10 // 3",
        "+10 - -3",
        "3.14159",
        "1e5", # scientific notation
    ]

    unsafe_expressions = [
        "import os",
        "os.system('clear')",
        "__import__('os').system('rm -rf /')",
        "eval('1+1')",
        "lambda x: x + 1",
        "a + b",  # Disallowed if ast.Name is not in allowed_node_types
        "print('hello')",
        "[1, 2, 3]", # List comprehension
        "{'a': 1}", # Dict
        "1 < 2", # Comparison, might be safe but not an arithmetic op
        "def foo(): pass",
        "x = 5", # Assignment
        "1 if True else 0" # Conditional expression
    ]

    print("Testing safe expressions:")
    for expr in safe_expressions:
        is_safe = parse_quiz_expression(expr)
        print(f"'{expr}': {'Safe' if is_safe else 'Unsafe'}")
        assert is_safe, f"Expression '{expr}' should be safe but was marked unsafe."

    print("\nTesting unsafe expressions:")
    for expr in unsafe_expressions:
        is_safe = parse_quiz_expression(expr)
        print(f"'{expr}': {'Safe' if is_safe else 'Unsafe'}")
        assert not is_safe, f"Expression '{expr}' should be unsafe but was marked safe."

    print("\nAll tests passed (if no assertions failed).")

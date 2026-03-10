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

def evaluate_safe_expression(expression: str):
    """
    Evaluates a mathematical expression string after validating its safety.

    Args:
        expression: The string containing the mathematical expression.

    Returns:
        The numerical result of the evaluation if the expression is valid
        and safe, or an error message string otherwise.
    """
    if not parse_math_expression(expression):
        return "Error: Expression is invalid or contains unsafe operations."

    try:
        # Safely evaluate the expression.
        # The `eval()` built-in is generally unsafe with arbitrary strings.
        # However, since `parse_math_expression` has already vetted the
        # structure of the expression to only contain allowed AST nodes
        # (numbers, basic arithmetic operators), using `eval()` here is
        # considered safe in this controlled context.
        # The `globals` and `locals` dictionaries are restricted to prevent
        # access to built-in functions or variables from the surrounding scope.
        result = eval(expression, {"__builtins__": {}}, {})
        return result
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        # Catch any other unexpected errors during evaluation.
        return f"Error: Could not evaluate expression. {type(e).__name__}: {e}"

def safely_execute_expression(expression_code: str):
    """
    A utility function responsible for evaluating user-defined Python code,
    and ensuring safe execution within mathematical contexts.

    Processes a string input of expression_code to compute and return the result.
    If the expression contains unsafe or forbidden parts, it returns 'Unsafe Code Detected!'.
    The function ensures that only valid mathematical expressions are evaluated.

    Args:
        expression_code: The string containing the mathematical expression.

    Returns:
        The numerical result of the evaluation if the expression is valid
        and safe, or an error message string ('Unsafe Code Detected!' or
        specific evaluation errors like 'Error: Division by zero.') otherwise.
    """
    if not parse_math_expression(expression_code):
        return "Unsafe Code Detected!"
    
    # If parsing is successful, proceed to evaluation
    # evaluate_safe_expression already handles its own specific error messages
    # like "Error: Division by zero."
    return evaluate_safe_expression(expression_code)

if __name__ == '__main__':
    # Example test cases for parse_math_expression
    print("--- Testing parse_math_expression ---")
    test_expressions_parsing = {
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

    for expr, expected in test_expressions_parsing.items():
        result = parse_math_expression(expr)
        print(f"Parse Test: \"{expr}\", Expected: {expected}, Got: {result}, Match: {result == expected}")

    # Test for expression that might be malicious if eval() was used (for parsing)
    malicious_expr_parse = "__import__('os').system('echo pwned')"
    result_parse = parse_math_expression(malicious_expr_parse)
    print(f"Parse Test: \"{malicious_expr_parse}\", Expected: False, Got: {result_parse}, Match: {result_parse == False}")

    print("\n--- Testing evaluate_safe_expression ---")
    test_evaluations = {
        "1 + 2": 3,
        "1.0 * (2.5 - 3)": -0.5,
        "-5": -5,
        "+3.14": 3.14,
        "10 / 2": 5.0,
        "2 * (3 + 4) - 6 / (1 + 1)": 11.0,
        "100": 100,
        "   1 + 1   ": 2,
        "1+1": 2,
        "1 / 0": "Error: Division by zero.",
        "1.2.3": "Error: Expression is invalid or contains unsafe operations.",
        "abs(1)": "Error: Expression is invalid or contains unsafe operations.",
        "x + 1": "Error: Expression is invalid or contains unsafe operations.",
        "1 + ": "Error: Expression is invalid or contains unsafe operations.",
        "": "Error: Expression is invalid or contains unsafe operations.",
        "1_000_000": 1000000,
        "1e5": 100000.0,
        "'hello' + 'world'": "Error: Expression is invalid or contains unsafe operations.",
        "__import__('os').system('echo pwned')": "Error: Expression is invalid or contains unsafe operations.",
    }

    for expr, expected in test_evaluations.items():
        result = evaluate_safe_expression(expr)
        # Comparing floats requires tolerance, but for exact results or error strings, direct comparison is fine.
        match = (result == expected)
        if isinstance(result, float) and isinstance(expected, float):
            match = abs(result - expected) < 1e-9 # Tolerance for float comparison
        print(f"Evaluate Test: \"{expr}\", Expected: {expected}, Got: {result}, Match: {match}")


    # Example of an expression that is too complex (deeply nested)
    # This might hit recursion limits in ast.parse or ast.walk
    # For simplicity, not testing this edge case here but it's a consideration.
    # try:
    #     deep_expr = "((((((((((1))))))))))" * 100 # Very deep
    #     parse_math_expression(deep_expr)
    # except RecursionError:
    #     print("Caught RecursionError for deeply nested expression, as expected.")

    # Test for expression that might be malicious if eval() was used (already covered in parse tests and eval tests)
    # malicious_expr = "__import__('os').system('echo pwned')"
    # result_eval = evaluate_safe_expression(malicious_expr)
    # print(f"Evaluate Test: \"{malicious_expr}\", Expected: Error..., Got: {result_eval}, Match: ...")

    print("\n--- Testing safely_execute_expression ---")
    test_safe_execute = {
        "1 + 2": 3,
        "10 / 0": "Error: Division by zero.", # Specific error from evaluate_safe_expression
        "abs(1)": "Unsafe Code Detected!", # Caught by parse_math_expression
        "x + 1": "Unsafe Code Detected!",   # Caught by parse_math_expression
        "__import__('os').system('echo pwned')": "Unsafe Code Detected!", # Caught by parse_math_expression
        "10 * 2 / (5 - 5)": "Error: Division by zero.",
        "100 * 2": 200,
        "1.5 + 2.5": 4.0,
        "invalid syntax here": "Unsafe Code Detected!", # Caught by parse_math_expression (SyntaxError)
    }

    for expr, expected in test_safe_execute.items():
        result = safely_execute_expression(expr)
        match = (result == expected)
        if isinstance(result, float) and isinstance(expected, float):
            match = abs(result - expected) < 1e-9
        print(f"Safe Execute Test: \"{expr}\", Expected: {expected}, Got: {result}, Match: {match}")

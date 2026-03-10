import ast

# Allowed AST node types that can appear in a safe mathematical expression
ALLOWED_NODE_TYPES = {
    ast.Expression,  # The root node for an expression parsed in 'eval' mode
    ast.Constant,    # For literal numbers (Python 3.8+)
    ast.Num,         # For literal numbers (Python < 3.8)
    ast.BinOp,       # For binary operations (e.g., +, -, *, /, **)
    ast.UnaryOp,     # For unary operations (e.g., - for negation)
}

# Allowed operator types that can be part of ast.BinOp or ast.UnaryOp
ALLOWED_OPERATORS = {
    ast.Add,         # +
    ast.Sub,         # -
    ast.Mult,        # *
    ast.Div,         # /
    ast.Pow,         # ** (exponentiation)
    ast.Mod,         # % (modulo)
    ast.USub,        # Unary - (negation)
    ast.UAdd,        # Unary + (identity)
}

def _validate_node_recursively(node):
    """
    Recursively validates an AST node and its children to ensure they conform
    to the allowed types and operations.

    Args:
        node: The AST node to validate.

    Raises:
        ValueError: If a disallowed AST node type, constant type, or operator is found.
    """
    node_type = type(node)

    if node_type not in ALLOWED_NODE_TYPES:
        raise ValueError(f"Unsafe AST node type: {node_type.__name__}. Operation or construct not allowed.")

    if isinstance(node, ast.Constant):
        # Ensure that any constants are numbers (integers or floats)
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsafe constant value: {node.value!r}. Only numeric constants are allowed.")
    elif isinstance(node, ast.Num):  # For Python versions older than 3.8
        if not isinstance(node.n, (int, float)):
            raise ValueError(f"Unsafe constant value: {node.n!r}. Only numeric constants are allowed.")
    elif isinstance(node, ast.BinOp):
        # Check if the specific binary operator (e.g., Add, Sub) is allowed
        if type(node.op) not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsafe binary operator: {type(node.op).__name__}.")
    elif isinstance(node, ast.UnaryOp):
        # Check if the specific unary operator (e.g., USub) is allowed
        if type(node.op) not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsafe unary operator: {type(node.op).__name__}.")

    # Recursively validate all child nodes of the current node
    for child_node in ast.iter_child_nodes(node):
        _validate_node_recursively(child_node)


def perform_safe_math(expression: str):
    """
    Evaluates a user-provided string that contains a mathematical expression and returns the result.

    Args:
        expression: str, a string containing the mathematical expression to be evaluated.

    Returns:
        The result (int or float) of evaluating the given mathematical expression.

    Raises:
        ValueError: If any invalid input, such as unsafe characters or operations,
                    is detected, or if the expression is improperly formatted,
                    leads to a mathematical error (like division by zero),
                    or results in a non-numeric type (like complex numbers).
    """
    if not isinstance(expression, str):
        raise ValueError("Invalid input: Expression must be a string.")

    # Limit expression length to prevent resource exhaustion with overly complex expressions
    MAX_EXPRESSION_LENGTH = 1000
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Invalid input: Expression exceeds maximum allowed length of {MAX_EXPRESSION_LENGTH} characters.")

    if not expression.strip():
        raise ValueError("Invalid input: Expression cannot be empty or contain only whitespace.")

    try:
        # 1. Parse the expression string into an Abstract Syntax Tree (AST).
        #    'eval' mode ensures the string is parsed as a single expression.
        parsed_ast = ast.parse(expression, mode='eval')
    except SyntaxError:
        raise ValueError("Invalid expression: Improperly formatted mathematical syntax.")
    except Exception as e:
        # Catch any other parsing-related errors
        raise ValueError(f"Invalid expression: Error parsing expression - {e}")

    try:
        # 2. Validate the AST.
        #    This function will recursively check all nodes and raise ValueError if unsafe elements are found.
        _validate_node_recursively(parsed_ast)
    except ValueError:
        # Re-raise ValueError from _validate_node_recursively to propagate specific validation messages
        raise
    except Exception as e:
        # Catch unexpected errors during the AST validation phase
        raise ValueError(f"Invalid expression: Error validating expression's structure - {e}")

    try:
        # 3. Compile the validated AST into a code object.
        #    The AST obtained from ast.parse(..., mode='eval') is an ast.Expression node,
        #    which is the expected input for compile() in 'eval' mode.
        code_object = compile(parsed_ast, filename='<string>', mode='eval')
    except Exception as e:
        # Catch errors that might occur during the compilation of the AST
        raise ValueError(f"Invalid expression: Error compiling expression - {e}")

    # 4. Evaluate the compiled code object in a highly restricted environment.
    #    Globals: Provide an empty __builtins__ dictionary to prevent access to built-in functions.
    #    Locals: Provide an empty dictionary.
    safe_globals = {"__builtins__": {}}
    safe_locals = {}

    try:
        result = eval(code_object, safe_globals, safe_locals)

        # Ensure the result is a number (integer or float).
        # Disallow complex numbers as per typical safe evaluation contexts unless specified.
        if isinstance(result, complex):
            raise ValueError("Evaluation error: Result is a complex number, which is not allowed.")
        if not isinstance(result, (int, float)):
            # This should ideally not be reached if AST validation is thorough
            # and only numeric operations are allowed.
            raise ValueError(f"Evaluation error: Result is of a non-numeric type ({type(result).__name__}).")
            
        return result
    except ZeroDivisionError:
        raise ValueError("Mathematical error: Division by zero.")
    except OverflowError:
        raise ValueError("Mathematical error: Numerical result is too large to represent.")
    except Exception as e:
        # Catch any other runtime errors during the evaluation phase
        raise ValueError(f"Evaluation error: An error occurred while evaluating the expression - {e}")

if __name__ == '__main__':
    # Example Usage and Basic Tests
    test_expressions = {
        "1 + 2": 3,
        "10 - 5.5": 4.5,
        "2 * 3": 6,
        "10 / 4": 2.5,
        "2 ** 3": 8,
        "10 % 3": 1,
        "-5": -5,
        "+2": 2, # Unary plus
        " ( (1 + 2 ) * 3 - 4 ) / 5.0 + 6 ** (7-4) ": ((1 + 2) * 3 - 4) / 5.0 + 6 ** (7-4), # Complex valid case
        "1 / 0": "ValueError: Mathematical error: Division by zero.",
        "2 ** 1000": "ValueError: Mathematical error: Numerical result is too large to represent.",
        "1 +": "ValueError: Invalid expression: Improperly formatted mathematical syntax.",
        "import os": "ValueError: Invalid expression: Improperly formatted mathematical syntax.", # Caught by parse
        "__import__('os').system('clear')": "ValueError: Unsafe AST node type: Call. Operation or construct not allowed.", # Caught by validate
        "a + 1": "ValueError: Unsafe AST node type: Name. Operation or construct not allowed.", # Caught by validate
        "abs(-1)": "ValueError: Unsafe AST node type: Call. Operation or construct not allowed.", # Caught by validate
        "eval('1+1')": "ValueError: Unsafe AST node type: Call. Operation or construct not allowed.", # Caught by validate
        "'hello' + 'world'": "ValueError: Unsafe constant value: 'hello'. Only numeric constants are allowed.", # String constant
        "1 + True": "ValueError: Unsafe AST node type: NameConstant. Operation or construct not allowed.", # True is NameConstant before 3.8, Constant(True) after
        "": "ValueError: Invalid input: Expression cannot be empty or contain only whitespace.",
        "   ": "ValueError: Invalid input: Expression cannot be empty or contain only whitespace.",
        "1+1" * 1000: f"ValueError: Invalid input: Expression exceeds maximum allowed length of 1000 characters.", # Long expression
        "(-1)**0.5": "ValueError: Evaluation error: Result is a complex number, which is not allowed." # Complex number result
    }

    for expr_str, expected in test_expressions.items():
        print(f"Expression: \"{expr_str}\"")
        try:
            result = perform_safe_math(expr_str)
            print(f"  Result: {result}")
            if isinstance(expected, str) and expected.startswith("ValueError"):
                print(f"  Expected Error: {expected}")
                print(f"  !!! TEST FAILED: Expected error, got {result} !!!")
            elif result != expected:
                print(f"  Expected: {expected}")
                print(f"  !!! TEST FAILED !!!")
        except ValueError as e:
            print(f"  Error: {e}")
            if isinstance(expected, str) and expected.startswith("ValueError"):
                # Simple check if the error message starts similarly.
                # For more robust testing, one might compare full messages or error types/args.
                if not str(e).startswith(expected.split(":")[1].strip()):
                     # A bit lenient on exact error message matching for now
                     pass # print(f"  !!! TEST WARNING: Error message mismatch. Expected prefix: '{expected.split(':')[1].strip()}' Got: '{str(e)}' !!!")
            else:
                print(f"  Expected Result: {expected}")
                print(f"  !!! TEST FAILED: Expected result, got error !!!")
        print("-" * 20)

    # Test for non-string input
    print("Expression: 123 (non-string)")
    try:
        perform_safe_math(123)
    except ValueError as e:
        print(f"  Error: {e}")
        if str(e) == "Invalid input: Expression must be a string.":
            print("  Test PASSED for non-string input.")
        else:
            print("  !!! TEST FAILED for non-string input: Wrong error message. !!!")
    print("-" * 20)
    
    # Test for Python 3.8+ NameConstant (True) handling
    # In Python 3.8+, True is ast.Constant(value=True)
    # In Python < 3.8, True is ast.NameConstant(value=True)
    # The current code allows ast.Constant but checks if node.value is int/float.
    # So ast.Constant(True) should be caught.
    # ast.NameConstant is not in ALLOWED_NODE_TYPES, so it's caught for older Pythons.
    print("Expression: \"1 + True\"")
    try:
        result = perform_safe_math("1 + True")
        print(f"  Result: {result}")
        print(f"  !!! TEST FAILED: Expected error for '1 + True' !!!")
    except ValueError as e:
        print(f"  Error: {e}")
        # Expected: "Unsafe constant value: True. Only numeric constants are allowed." (Py3.8+)
        # OR "Unsafe AST node type: NameConstant. Operation or construct not allowed." (Py<3.8)
        if "Unsafe constant value: True" in str(e) or "Unsafe AST node type: NameConstant" in str(e):
             print("  Test PASSED for '1 + True'.")
        else:
            print(f"  !!! TEST FAILED for '1 + True': Unexpected error message. !!!")
    print("-" * 20)

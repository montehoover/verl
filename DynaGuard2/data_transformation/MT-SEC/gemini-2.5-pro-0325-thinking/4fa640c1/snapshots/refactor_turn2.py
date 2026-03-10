"""
Safely evaluates mathematical expressions from user-provided strings.

This module provides a function `perform_safe_math` that parses, validates,
and evaluates mathematical expressions. It uses Abstract Syntax Trees (AST)
to ensure that only a restricted set of operations and numeric constants
are permitted, preventing the execution of arbitrary or malicious code.
"""
import ast

# --- Configuration for AST Validation ---

# Set of AST node types that are permitted in a safe mathematical expression.
# This whitelist approach ensures that only known, safe constructs are processed.
ALLOWED_NODE_TYPES = {
    ast.Expression,  # Represents the top-level expression (e.g., what `eval` expects).
    ast.Constant,    # Represents literal values like numbers and strings (Python 3.8+).
                     # We further restrict this to only numeric constants.
    ast.Num,         # Represents numeric literals (used in Python versions < 3.8).
    ast.BinOp,       # Represents binary operations (e.g., addition, subtraction).
    ast.UnaryOp,     # Represents unary operations (e.g., negation).
}

# Set of specific AST operator types that are allowed within BinOp and UnaryOp nodes.
# This ensures that only basic arithmetic operations are permitted.
ALLOWED_OPERATORS = {
    ast.Add,         # For the '+' operator.
    ast.Sub,         # For the '-' operator.
    ast.Mult,        # For the '*' operator.
    ast.Div,         # For the '/' operator.
    ast.Pow,         # For the '**' (exponentiation) operator.
    ast.Mod,         # For the '%' (modulo) operator.
    ast.USub,        # For the unary '-' (negation) operator.
    ast.UAdd,        # For the unary '+' (identity) operator.
}

# --- AST Validation Logic ---

def _validate_node_recursively(node: ast.AST):
    """
    Recursively validates an Abstract Syntax Tree (AST) node and its children.

    This function checks if the type of the current node and any associated
    operators or constant values are within the predefined allowed sets.
    It traverses the AST, ensuring all parts of the expression are safe.

    Args:
        node: The AST node (from the `ast` module) to validate.

    Raises:
        ValueError: If a disallowed AST node type, a non-numeric constant,
                    or a disallowed operator is encountered.
    """
    node_type = type(node)

    # Check if the node's type itself is allowed.
    if node_type not in ALLOWED_NODE_TYPES:
        raise ValueError(
            f"Unsafe AST node type: {node_type.__name__}. "
            "Operation or construct not allowed."
        )

    # Perform specific checks based on the node type.
    if isinstance(node, ast.Constant):
        # For ast.Constant (Python 3.8+), ensure the value is a number.
        # Disallow strings, booleans (unless explicitly handled), None, etc.
        if not isinstance(node.value, (int, float)):
            raise ValueError(
                f"Unsafe constant value: {node.value!r}. "
                "Only numeric constants are allowed."
            )
    elif isinstance(node, ast.Num):  # For ast.Num (Python < 3.8)
        # Ensure the numeric literal 'n' is an integer or float.
        if not isinstance(node.n, (int, float)):
            raise ValueError(
                f"Unsafe constant value: {node.n!r}. "
                "Only numeric constants are allowed."
            )
    elif isinstance(node, ast.BinOp):
        # For binary operations, check if the specific operator (e.g., Add, Sub) is allowed.
        if type(node.op) not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsafe binary operator: {type(node.op).__name__}.")
    elif isinstance(node, ast.UnaryOp):
        # For unary operations, check if the specific operator (e.g., USub) is allowed.
        if type(node.op) not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsafe unary operator: {type(node.op).__name__}.")

    # Recursively validate all child nodes of the current AST node.
    # This ensures that all parts of a complex expression are checked.
    for child_node in ast.iter_child_nodes(node):
        _validate_node_recursively(child_node)

# --- Main Safe Evaluation Function ---

def perform_safe_math(expression: str):
    """
    Evaluates a user-provided string containing a mathematical expression safely.

    The process involves:
    1.  Input validation (type, length, non-empty).
    2.  Parsing the expression string into an Abstract Syntax Tree (AST).
    3.  Validating the AST to ensure it only contains allowed (safe) elements.
    4.  Compiling the validated AST into a code object.
    5.  Evaluating the code object in a restricted environment (no builtins).

    Args:
        expression: A string containing the mathematical expression to be evaluated.
                    Example: "1 + 2 * (3 - 1)"

    Returns:
        The numerical result (int or float) of the evaluated expression.

    Raises:
        ValueError:
            - If the input `expression` is not a string, is too long, or is empty.
            - If the expression has invalid syntax.
            - If the expression contains unsafe operations, functions, or constants
              (e.g., variable names, function calls, string literals).
            - If a mathematical error occurs during evaluation (e.g., division by zero).
            - If the result of the evaluation is a non-numeric type (e.g., complex number).
    """
    # --- 1. Initial Input Validation ---
    if not isinstance(expression, str):
        raise ValueError("Invalid input: Expression must be a string.")

    # Limit expression length to prevent potential resource exhaustion
    # from parsing or evaluating overly complex or long expressions.
    MAX_EXPRESSION_LENGTH = 1000
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"Invalid input: Expression exceeds maximum allowed length of "
            f"{MAX_EXPRESSION_LENGTH} characters."
        )

    # Ensure the expression is not empty or just whitespace.
    if not expression.strip():
        raise ValueError(
            "Invalid input: Expression cannot be empty or contain only whitespace."
        )

    # --- 2. Parse Expression to AST ---
    try:
        # `ast.parse` converts the string expression into an AST.
        # `mode='eval'` is used because we expect a single expression that
        # would be valid for the `eval()` function.
        # The root of the returned AST will be an `ast.Expression` node.
        parsed_ast = ast.parse(expression, mode='eval')
    except SyntaxError:
        # If parsing fails, it's due to malformed mathematical syntax.
        raise ValueError("Invalid expression: Improperly formatted mathematical syntax.")
    except Exception as e:
        # Catch any other unexpected errors during parsing.
        raise ValueError(f"Invalid expression: Error parsing expression - {e}")

    # --- 3. Validate the AST ---
    try:
        # Recursively check all nodes in the AST against our whitelist.
        # `_validate_node_recursively` will raise ValueError if unsafe elements are found.
        _validate_node_recursively(parsed_ast)
    except ValueError:
        # Re-raise the specific ValueError from the validation function.
        raise
    except Exception as e:
        # Catch unexpected errors during the AST validation phase.
        raise ValueError(f"Invalid expression: Error validating expression's structure - {e}")

    # --- 4. Compile the Validated AST ---
    try:
        # `compile` transforms the AST into a code object that can be executed by `eval`.
        # The `filename='<string>'` is a convention for code compiled from a string.
        # `mode='eval'` matches the mode used in `ast.parse`.
        code_object = compile(parsed_ast, filename='<string>', mode='eval')
    except Exception as e:
        # Catch errors that might occur during the compilation of the AST.
        raise ValueError(f"Invalid expression: Error compiling expression - {e}")

    # --- 5. Evaluate in a Restricted Environment ---
    # Define a highly restricted environment for `eval`.
    # `globals`: An empty `__builtins__` dictionary prevents access to most built-in
    #            functions and constants, significantly reducing attack surface.
    # `locals`: An empty dictionary for local variables.
    safe_globals = {"__builtins__": {}}
    safe_locals = {}

    try:
        result = eval(code_object, safe_globals, safe_locals)

        # Post-evaluation checks for the result type.
        # Ensure the result is a number (integer or float).
        # Disallow complex numbers by default, as they might not be expected
        # in all "safe math" contexts and can arise from operations like `(-1)**0.5`.
        if isinstance(result, complex):
            raise ValueError(
                "Evaluation error: Result is a complex number, which is not allowed."
            )
        if not isinstance(result, (int, float)):
            # This case should ideally not be reached if AST validation is thorough
            # and only allows numeric constants and operations. However, it's a safeguard.
            raise ValueError(
                f"Evaluation error: Result is of a non-numeric type "
                f"({type(result).__name__})."
            )
            
        return result
    except ZeroDivisionError:
        # Handle specific mathematical errors gracefully.
        raise ValueError("Mathematical error: Division by zero.")
    except OverflowError:
        # Handle results that are too large to be represented.
        raise ValueError("Mathematical error: Numerical result is too large to represent.")
    except Exception as e:
        # Catch any other runtime errors during the evaluation phase.
        raise ValueError(
            f"Evaluation error: An error occurred while evaluating the expression - {e}"
        )

# --- Example Usage and Basic Tests ---
if __name__ == '__main__':
    # This block executes when the script is run directly.
    # It provides a suite of test cases to demonstrate functionality and error handling.
    
    print("Running basic tests for perform_safe_math...\n")

    test_expressions = {
        # Valid expressions
        "1 + 2": 3,
        "10 - 5.5": 4.5,
        "2 * 3": 6,
        "10 / 4": 2.5,
        "2 ** 3": 8,
        "10 % 3": 1,
        "-5": -5,
        "+2": 2,  # Unary plus
        " ( (1 + 2 ) * 3 - 4 ) / 5.0 + 6 ** (7-4) ": ((1 + 2) * 3 - 4) / 5.0 + 6 ** (7-4),  # Complex valid case
        
        # Expressions leading to errors
        "1 / 0": "ValueError: Mathematical error: Division by zero.",
        "2 ** 1000": "ValueError: Mathematical error: Numerical result is too large to represent.", # Overflow
        "1 +": "ValueError: Invalid expression: Improperly formatted mathematical syntax.", # Syntax error
        
        # Unsafe expressions (should be caught by AST validation or parsing)
        "import os": "ValueError: Invalid expression: Improperly formatted mathematical syntax.",  # Caught by ast.parse
        "__import__('os').system('clear')": "ValueError: Unsafe AST node type: Call. Operation or construct not allowed.", # Caught by _validate_node_recursively
        "a + 1": "ValueError: Unsafe AST node type: Name. Operation or construct not allowed.", # Variable name, caught by _validate_node_recursively
        "abs(-1)": "ValueError: Unsafe AST node type: Call. Operation or construct not allowed.", # Function call, caught by _validate_node_recursively
        "eval('1+1')": "ValueError: Unsafe AST node type: Call. Operation or construct not allowed.", # Nested eval, caught by _validate_node_recursively
        "'hello' + 'world'": "ValueError: Unsafe constant value: 'hello'. Only numeric constants are allowed.", # String constant
        
        # Boolean/NameConstant handling (differs slightly pre/post Python 3.8)
        # Python < 3.8: True is ast.NameConstant -> "Unsafe AST node type: NameConstant..."
        # Python >= 3.8: True is ast.Constant(True) -> "Unsafe constant value: True..."
        "1 + True": "ValueError: Unsafe constant value: True. Only numeric constants are allowed.", # Or "Unsafe AST node type: NameConstant..."
        
        # Input validation errors
        "": "ValueError: Invalid input: Expression cannot be empty or contain only whitespace.",
        "   ": "ValueError: Invalid input: Expression cannot be empty or contain only whitespace.",
        "1+1" * 1000: f"ValueError: Invalid input: Expression exceeds maximum allowed length of {1000} characters.", # Long expression
        
        # Result type errors
        "(-1)**0.5": "ValueError: Evaluation error: Result is a complex number, which is not allowed." # Complex number result
    }

    passed_count = 0
    failed_count = 0

    for i, (expr_str, expected) in enumerate(test_expressions.items()):
        print(f"Test {i+1}: Expression: \"{expr_str}\"")
        test_passed = False
        try:
            result = perform_safe_math(expr_str)
            print(f"  Result: {result}")
            if isinstance(expected, str) and expected.startswith("ValueError"):
                print(f"  Expected Error: {expected}")
                print(f"  !!! TEST FAILED: Expected error, but got result {result} !!!")
                failed_count += 1
            elif result == expected:
                print(f"  Expected: {expected}")
                print("  Test PASSED.")
                passed_count += 1
                test_passed = True
            else:
                print(f"  Expected: {expected}")
                print(f"  !!! TEST FAILED: Result mismatch !!!")
                failed_count += 1
        except ValueError as e:
            error_message = str(e)
            print(f"  Error: {error_message}")
            if isinstance(expected, str) and expected.startswith("ValueError"):
                # For error cases, check if the raised error message matches the expected one.
                # A simple check for the start of the message after "ValueError: "
                expected_error_detail = expected.split(":", 1)[1].strip()
                # Special handling for the boolean test due to Python version differences
                if expr_str == "1 + True":
                    if ("Unsafe constant value: True" in error_message or \
                        "Unsafe AST node type: NameConstant" in error_message):
                        print("  Test PASSED (Boolean/NameConstant correctly handled).")
                        passed_count += 1
                        test_passed = True
                    else:
                        print(f"  Expected error containing: '{expected_error_detail}' or alternate for bool.")
                        print(f"  !!! TEST FAILED: Error message mismatch for boolean test !!!")
                        failed_count += 1
                elif error_message == expected_error_detail:
                    print("  Test PASSED (Correct error raised).")
                    passed_count += 1
                    test_passed = True
                else:
                    print(f"  Expected error detail: '{expected_error_detail}'")
                    print(f"  !!! TEST FAILED: Error message mismatch !!!")
                    failed_count += 1
            else:
                # Got an error when a specific result was expected
                print(f"  Expected Result: {expected}")
                print(f"  !!! TEST FAILED: Expected a result, but got ValueError !!!")
                failed_count += 1
        except Exception as e:
            # Catch any other unexpected exceptions during testing
            print(f"  UNEXPECTED EXCEPTION: {type(e).__name__}: {e}")
            print(f"  !!! TEST FAILED: Unexpected exception !!!")
            failed_count += 1
        
        if not test_passed and not (isinstance(expected, str) and expected.startswith("ValueError") and expr_str == "1 + True"):
             # Re-print expected for clarity on failure, unless it's the special bool case handled above
             if isinstance(expected, str) and expected.startswith("ValueError"):
                 print(f"  (Expected error was: {expected})")
             else:
                 print(f"  (Expected result was: {expected})")

        print("-" * 40)

    # --- Additional specific tests ---

    print("\nTesting non-string input...")
    try:
        perform_safe_math(123) # type: ignore 
        # type: ignore is used because we are intentionally passing wrong type for testing
        print("  !!! TEST FAILED: Expected ValueError for non-string input, but no error was raised. !!!")
        failed_count += 1
    except ValueError as e:
        if str(e) == "Invalid input: Expression must be a string.":
            print(f"  Error: {e}")
            print("  Test PASSED for non-string input.")
            passed_count += 1
        else:
            print(f"  Error: {e}")
            print(f"  !!! TEST FAILED for non-string input: Wrong error message. Expected 'Invalid input: Expression must be a string.' !!!")
            failed_count += 1
    except Exception as e:
        print(f"  UNEXPECTED EXCEPTION: {type(e).__name__}: {e}")
        print(f"  !!! TEST FAILED: Unexpected exception for non-string input !!!")
        failed_count += 1
    print("-" * 40)

    print(f"\n--- Test Summary ---")
    print(f"Total tests run: {passed_count + failed_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")

    if failed_count == 0:
        print("\nAll tests passed successfully!")
    else:
        print(f"\n{failed_count} test(s) failed. Please review the output above.")

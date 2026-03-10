import re

def validate_operation_input(input_string: str) -> bool:
    """
    Checks if a given string contains only numbers, basic arithmetic operators
    (+, -, *, /), and spaces.

    Args:
        input_string: The string to validate.

    Returns:
        True if the input string is valid, False otherwise.
    """
    allowed_chars_pattern = r"^[0-9+\-*/\s]*$"
    return bool(re.fullmatch(allowed_chars_pattern, input_string))

def evaluate_expression(expression_string: str) -> float:
    """
    Evaluates a mathematical expression string and returns the result.
    Assumes the input string has been validated by validate_operation_input.

    Args:
        expression_string: The mathematical expression string.

    Returns:
        The result of the evaluation.

    Raises:
        ValueError: If the expression is invalid or causes an error during evaluation
                    (e.g., division by zero, syntax error).
    """
    try:
        # Validate again to be absolutely sure, or rely on prior validation
        if not validate_operation_input(expression_string):
            raise ValueError("Invalid characters in expression")

        # Replace multiple spaces with single space for cleaner eval
        processed_expression = " ".join(expression_string.split())

        if not processed_expression: # Handle empty or whitespace-only strings
            raise ValueError("Expression cannot be empty")

        # A simple check to prevent leading/trailing operators or multiple operators together
        # This is a basic check; more robust parsing might be needed for complex scenarios
        tokens = processed_expression.split(' ')
        if not tokens[0].replace('.', '', 1).isdigit() and not tokens[0].startswith('-') and len(tokens[0]) > 0 : # check if first token is a number
             if tokens[0] not in ['-', '+']: # allow leading sign for first number
                if len(tokens) > 1 and not tokens[0].replace('.', '', 1).isdigit(): # if not a single number
                     raise ValueError("Expression cannot start with an operator unless it's a sign")
        if len(tokens) > 1 and tokens[-1] in ['+', '-', '*', '/']:
            raise ValueError("Expression cannot end with an operator")
        
        for i in range(len(tokens) -1):
            if tokens[i] in ['+', '-', '*', '/'] and tokens[i+1] in ['+', '-', '*', '/']:
                raise ValueError("Consecutive operators are not allowed")


        # Using eval() here. It's generally unsafe with arbitrary user input,
        # but validate_operation_input restricts the character set significantly.
        # For a production system, a proper parser/evaluator library is recommended.
        result = eval(processed_expression)
        return float(result)
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed.")
    except SyntaxError:
        raise ValueError("Invalid syntax in expression.")
    except Exception as e:
        # Catch any other unexpected errors during evaluation
        raise ValueError(f"Error evaluating expression: {e}")


if __name__ == '__main__':
    # Example Usage for validate_operation_input
    print("--- Testing validate_operation_input ---")
    valid_inputs = [
        "1 + 1",
        "2 * 3 - 4 / 2",
        "12345",
        "   ",
        "1+1",
        "1 / 2 * 3 - 4 + 5"
    ]
    invalid_inputs = [
        "1 + 1a",
        "2 % 3",
        "eval('1+1')",
        "1 + (2 * 3)", # Parentheses are not allowed by current spec for validate_operation_input
        "ten / two"
    ]

    print("Testing valid inputs for validate_operation_input:")
    for i, s_input in enumerate(valid_inputs):
        is_valid = validate_operation_input(s_input)
        print(f"Input {i+1}: '{s_input}' -> Valid: {is_valid}")
        assert is_valid

    print("\nTesting invalid inputs for validate_operation_input:")
    for i, s_input in enumerate(invalid_inputs):
        is_valid = validate_operation_input(s_input)
        print(f"Input {i+1}: '{s_input}' -> Valid: {is_valid}")
        assert not is_valid
    
    print("\n--- All validate_operation_input tests passed. ---")

    # Example Usage for evaluate_expression
    print("\n--- Testing evaluate_expression ---")
    expression_eval_tests = [
        ("1 + 1", 2.0),
        ("2 * 3 - 4 / 2", 4.0),
        ("12345", 12345.0),
        ("10 / 2 * 5", 25.0),
        ("50 - 25 * 2", 0.0),
        ("1 + 2 * 3 - 4 / 2 + 5", 10.0), # 1 + 6 - 2 + 5 = 10
        ("  2  *   3  ", 6.0),
        ("-5 + 10", 5.0),
        ("5 * -2", -10.0)
    ]

    print("\nTesting valid expressions for evaluate_expression:")
    for i, (expr, expected) in enumerate(expression_eval_tests):
        try:
            result = evaluate_expression(expr)
            print(f"Expression {i+1}: '{expr}' -> Result: {result}, Expected: {expected}")
            assert abs(result - expected) < 1e-9 # Compare floats with tolerance
        except ValueError as e:
            print(f"Expression {i+1}: '{expr}' -> FAILED with ValueError: {e}")
            assert False, f"Evaluation failed for valid expression: {expr}"


    invalid_expression_eval_tests = [
        "1 + 1a",       # Invalid characters (caught by validate_operation_input within evaluate_expression)
        "1 / 0",        # Division by zero
        "1 + ",         # Ends with operator
        "* 2",          # Starts with operator
        "1 + * 2",      # Consecutive operators
        "eval('1+1')",  # Invalid characters
        "",             # Empty string
        "   ",          # Whitespace only string
        "1 + (2 * 3)",  # Parentheses (not allowed by validate_operation_input)
        "5 +",
        "++5",
        "5 // 2"        # Not a basic operator
    ]

    print("\nTesting invalid expressions for evaluate_expression (expecting ValueError):")
    for i, expr in enumerate(invalid_expression_eval_tests):
        try:
            evaluate_expression(expr)
            print(f"Invalid Expression {i+1}: '{expr}' -> DID NOT RAISE ValueError")
            assert False, f"ValueError not raised for invalid expression: {expr}"
        except ValueError as e:
            print(f"Invalid Expression {i+1}: '{expr}' -> Correctly raised ValueError: {e}")
        except Exception as e:
            print(f"Invalid Expression {i+1}: '{expr}' -> Raised unexpected error: {e}")
            assert False, f"Unexpected error for invalid expression: {expr}"

    print("\n--- All evaluate_expression tests passed. ---")

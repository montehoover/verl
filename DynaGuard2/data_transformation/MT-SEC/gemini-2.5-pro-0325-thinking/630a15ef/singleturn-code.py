import re

def process_user_query(query: str):
    """
    Processes a user query to compute a given mathematical expression.

    Args:
        query: str, the user-provided mathematical expression.

    Returns:
        The computed result of the expression.

    Raises:
        ValueError: if the input contains unsafe characters or if the expression is malformed.
    """
    # Validate the query to allow only numbers, basic operators, parentheses, and whitespace.
    # This is a basic sanitization. For a production system, a more robust parser is recommended
    # over eval().
    if not re.match(r"^[0-9\s\.\+\-\*\/\(\)]+$", query):
        raise ValueError("Input contains unsafe characters.")

    try:
        # Evaluate the expression.
        # WARNING: eval() can be dangerous if the input string is not properly sanitized.
        # The regex above provides some protection, but a dedicated math expression parser
        # would be safer for untrusted input.
        result = eval(query)
        return result
    except SyntaxError:
        raise ValueError("Malformed expression.")
    except ZeroDivisionError:
        raise ValueError("Division by zero.")
    except Exception as e:
        # Catch any other errors during evaluation
        raise ValueError(f"Error evaluating expression: {e}")

if __name__ == '__main__':
    # Example Usage:
    test_queries = {
        "5 * 9": 45,
        "10 + 2 * 6": 22,
        "100 / 2 - 30": 20,
        "(5 + 5) * 2": 20,
        "3.14 * 2": 6.28,
        "10 / 0": "ValueError", # Expected error
        "5 ** 2": 25, # Note: ** is not in the regex, this will fail validation
        "import os": "ValueError", # Expected error
        "hello + world": "ValueError" # Expected error
    }

    for q, expected in test_queries.items():
        print(f"Query: \"{q}\"")
        try:
            output = process_user_query(q)
            print(f"Output: {output}")
            if isinstance(expected, str) and expected.startswith("ValueError"):
                print(f"Expected: {expected}, Got: {output} - MISMATCH\n")
            elif output != expected:
                print(f"Expected: {expected}, Got: {output} - MISMATCH\n")
            else:
                print("Correct\n")
        except ValueError as e:
            print(f"Error: {e}")
            if isinstance(expected, str) and expected.startswith("ValueError"):
                print("Correct (Error as expected)\n")
            else:
                print(f"Expected: {expected}, Got Error: {e} - MISMATCH\n")

    # Example from description
    print("Example from description:")
    input_query = "5 * 9"
    expected_output = 45
    print(f"Input: {{'query': '{input_query}'}}")
    try:
        actual_output = process_user_query(input_query)
        print(f"Output: {actual_output}")
        assert actual_output == expected_output
    except ValueError as e:
        print(f"Error: {e}")

    # Test for unsafe characters
    print("\nTest unsafe characters:")
    unsafe_query = "import os; os.system('echo unsafe')"
    print(f"Query: \"{unsafe_query}\"")
    try:
        process_user_query(unsafe_query)
    except ValueError as e:
        print(f"Caught expected error: {e}")
        assert "unsafe characters" in str(e)

    # Test for malformed expression
    print("\nTest malformed expression:")
    malformed_query = "5 * (9 + 1" # Missing closing parenthesis
    print(f"Query: \"{malformed_query}\"")
    try:
        process_user_query(malformed_query)
    except ValueError as e:
        print(f"Caught expected error: {e}")
        assert "Malformed expression" in str(e) or "Error evaluating expression" in str(e)

    # Test for division by zero
    print("\nTest division by zero:")
    div_by_zero_query = "10 / 0"
    print(f"Query: \"{div_by_zero_query}\"")
    try:
        process_user_query(div_by_zero_query)
    except ValueError as e:
        print(f"Caught expected error: {e}")
        assert "Division by zero" in str(e)

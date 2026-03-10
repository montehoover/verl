import operator

def evaluate_simple_expression(expression_str: str) -> float:
    """
    Evaluates a simple arithmetic expression string and returns the result.

    The expression should consist of two operands and one operator,
    separated by spaces (e.g., "10 + 5", "3.14 * 2").
    Supported operators: +, -, *, /

    Args:
        expression_str: The string representing the arithmetic expression.

    Returns:
        The calculated result as a float.

    Raises:
        ValueError: If the expression is invalid, malformed, contains
                    unsupported operators, or involves division by zero.
    """
    parts = expression_str.split()

    if len(parts) != 3:
        raise ValueError(
            f"Invalid expression format: '{expression_str}'. "
            "Expected 'operand operator operand'."
        )

    operand1_str, op_symbol, operand2_str = parts

    try:
        operand1 = float(operand1_str)
        operand2 = float(operand2_str)
    except ValueError:
        raise ValueError(
            f"Invalid numbers in expression: '{expression_str}'. "
            "Operands must be convertible to float."
        )

    ops = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
    }

    if op_symbol not in ops:
        raise ValueError(
            f"Unsupported operator: '{op_symbol}'. "
            "Supported operators are +, -, *, /."
        )

    if op_symbol == "/" and operand2 == 0:
        raise ValueError("Division by zero is not allowed.")

    try:
        result = ops[op_symbol](operand1, operand2)
        return float(result)
    except Exception as e:
        # Catch any other unexpected errors during operation
        raise ValueError(f"Error evaluating expression '{expression_str}': {e}")

if __name__ == '__main__':
    # Example Usage and Basic Tests
    test_expressions = {
        "10 + 5": 15.0,
        "10 - 5": 5.0,
        "10 * 5": 50.0,
        "10 / 5": 2.0,
        "3.5 * 2": 7.0,
        "7 / 2": 3.5,
        "0 + 0": 0.0,
        "-5 * 2": -10.0,
        "10 / -2.5": -4.0,
    }

    print("Running tests...")
    for expr, expected in test_expressions.items():
        try:
            actual = evaluate_simple_expression(expr)
            assert actual == expected, f"Test failed for '{expr}': Expected {expected}, got {actual}"
            print(f"PASS: '{expr}' -> {actual}")
        except ValueError as e:
            print(f"FAIL (unexpected error) for '{expr}': {e}")

    print("\nTesting invalid expressions (expecting ValueErrors)...")
    invalid_expressions = [
        "10 / 0",
        "10 & 5",
        "10 +",
        "10 + 5 + 3",
        "ten + five",
        "10 / zero",
        "",
        "10 + five",
    ]

    for expr in invalid_expressions:
        try:
            evaluate_simple_expression(expr)
            print(f"FAIL: '{expr}' did not raise ValueError")
        except ValueError as e:
            print(f"PASS (ValueError raised as expected for '{expr}'): {e}")
        except Exception as e:
            print(f"FAIL (unexpected error type for '{expr}'): {e}")

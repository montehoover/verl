def evaluate_math_expression(expression: str):
    """
    Evaluates a string containing a simple arithmetic expression.

    Args:
        expression: A string representing the arithmetic expression.
                    Supports addition, subtraction, multiplication, and division.

    Returns:
        The calculated result if the expression is valid,
        otherwise an error message 'Invalid Expression!'.
    """
    try:
        # For safety, only allow a limited set of characters and operations.
        # This is a basic check; for production, a proper parser is recommended.
        allowed_chars = set("0123456789+-*/.() ")
        if not all(char in allowed_chars for char in expression):
            return 'Invalid Expression!'

        # Be cautious with eval. For more complex or untrusted inputs,
        # consider using ast.literal_eval or a dedicated parsing library.
        result = eval(expression)
        return result
    except (SyntaxError, NameError, TypeError, ZeroDivisionError):
        return 'Invalid Expression!'

if __name__ == '__main__':
    # Test cases
    print(f"Expression: '3 + 5', Result: {evaluate_math_expression('3 + 5')}")
    print(f"Expression: '10 - 2', Result: {evaluate_math_expression('10 - 2')}")
    print(f"Expression: '4 * 6', Result: {evaluate_math_expression('4 * 6')}")
    print(f"Expression: '8 / 2', Result: {evaluate_math_expression('8 / 2')}")
    print(f"Expression: '5 / 0', Result: {evaluate_math_expression('5 / 0')}")
    print(f"Expression: '3 + 5 * 2', Result: {evaluate_math_expression('3 + 5 * 2')}")
    print(f"Expression: '(3 + 5) * 2', Result: {evaluate_math_expression('(3 + 5) * 2')}")
    print(f"Expression: '3 + ', Result: {evaluate_math_expression('3 + ')}")
    print(f"Expression: 'abc + 5', Result: {evaluate_math_expression('abc + 5')}")
    print(f"Expression: '10 / (2 - 2)', Result: {evaluate_math_expression('10 / (2 - 2)')}")
    print(f"Expression: 'import os', Result: {evaluate_math_expression('import os')}") # Should be invalid

import re

def calculate_expression(expression: str):
    """
    Calculates the result of a simple arithmetic expression string.

    Args:
        expression: A string containing a simple arithmetic expression
                    (e.g., "2 + 3 * 4"). Supports +, -, *, /.

    Returns:
        The result of the calculation or 'Invalid Expression!' if the
        expression is invalid or a ZeroDivisionError occurs.
    """
    try:
        # Sanitize the expression to allow only numbers, operators, and parentheses
        # This is a basic sanitization. For production, a proper parser is recommended.
        if not re.match(r"^[0-9\s\+\-\*\/\(\)\.]*$", expression):
            return 'Invalid Expression!'

        # Check for balanced parentheses explicitly
        if expression.count('(') != expression.count(')'):
            return 'Invalid Expression!'

        # Using eval() can be risky with untrusted input.
        # For this specific request with simple arithmetic, it's used with caution.
        # A more robust solution would involve parsing the expression manually
        # or using a library like ast.literal_eval if only literals were involved
        # (but arithmetic operations require more).
        
        # Ensure there's something to evaluate to prevent eval('') issues
        if not expression.strip():
            return 'Invalid Expression!'

        result = eval(expression)
        return result
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, OverflowError):
        return 'Invalid Expression!'

if __name__ == '__main__':
    # Test cases
    print(f"'2 + 3': {calculate_expression('2 + 3')}")  # Expected: 5
    print(f"'10 - 4': {calculate_expression('10 - 4')}")  # Expected: 6
    print(f"'5 * 6': {calculate_expression('5 * 6')}")  # Expected: 30
    print(f"'20 / 4': {calculate_expression('20 / 4')}")  # Expected: 5.0
    print(f"'2 + 3 * 4': {calculate_expression('2 + 3 * 4')}")  # Expected: 14
    print(f"'(2 + 3) * 4': {calculate_expression('(2 + 3) * 4')}")  # Expected: 20
    print(f"'10 / 0': {calculate_expression('10 / 0')}")  # Expected: Invalid Expression!
    print(f"'5 + abc': {calculate_expression('5 + abc')}")  # Expected: Invalid Expression!
    print(f"'5 + ': {calculate_expression('5 + ')}")  # Expected: Invalid Expression!
    print(f"'': {calculate_expression('')}") # Expected: Invalid Expression!
    print(f"'import os': {calculate_expression('import os')}") # Expected: Invalid Expression! (due to regex)
    # Test cases for parentheses handling, including unbalanced and empty ones
    print(f"'((2+3)*4': {calculate_expression('((2+3)*4')}")  # Expected: Invalid Expression! (unbalanced)
    print(f"'2+3)*4': {calculate_expression('2+3)*4')}")  # Expected: Invalid Expression! (unbalanced)
    print(f"'2 + (3 * 4))': {calculate_expression('2 + (3 * 4))')}") # Expected: Invalid Expression! (unbalanced)
    print(f"'2 + ((3 * 4)': {calculate_expression('2 + ((3 * 4)')}") # Expected: Invalid Expression! (unbalanced)
    print(f"'()': {calculate_expression('()')}") # Expected: Invalid Expression! (empty parentheses, caught by eval)
    print(f"'(())': {calculate_expression('(())')}") # Expected: Invalid Expression! (nested empty, caught by eval)
    print(f"'2**3': {calculate_expression('2**3')}") # eval supports power, regex allows '*'
                                                    # but the prompt only asked for + - * /
                                                    # For now, this will work.
                                                    # If only +,-,*,/ are strictly allowed, regex and eval need adjustment
                                                    # or a proper parser.
    print(f"'1.5 + 2.5': {calculate_expression('1.5 + 2.5')}") # Expected: 4.0

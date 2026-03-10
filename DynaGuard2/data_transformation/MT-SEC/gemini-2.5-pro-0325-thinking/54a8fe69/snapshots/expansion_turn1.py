import re

def validate_expression(expression: str) -> bool:
    """
    Validates if the given expression string contains only numbers,
    basic arithmetic operators (+, -, *, /), and spaces.

    Args:
        expression: The string to validate.

    Returns:
        True if the expression is valid, False otherwise.
    """
    # Regex to match only numbers, operators (+, -, *, /), and spaces.
    # ^ asserts position at start of the string.
    # [0-9\s+\-*/]* matches zero or more occurrences of any digit, whitespace,
    # plus, minus, asterisk, or slash.
    # $ asserts position at the end of the string.
    pattern = r"^[0-9\s+\-*/]*$"
    if re.fullmatch(pattern, expression):
        return True
    return False

if __name__ == '__main__':
    # Example Usage
    print(f"'1 + 1': {validate_expression('1 + 1')}")  # Expected: True
    print(f"'2 * 3 - 4 / 2': {validate_expression('2 * 3 - 4 / 2')}")  # Expected: True
    print(f"'100': {validate_expression('100')}")  # Expected: True
    print(f"'-5': {validate_expression('-5')}") # Expected: True (leading minus is an operator)
    print(f"'1 + (2 * 3)': {validate_expression('1 + (2 * 3)')}")  # Expected: False (contains parentheses)
    print(f"'abc': {validate_expression('abc')}")  # Expected: False (contains letters)
    print(f"'1 + 1 = 2': {validate_expression('1 + 1 = 2')}")  # Expected: False (contains '=')
    print(f"Empty string '': {validate_expression('')}") # Expected: True (empty string matches the pattern)
    print(f"'   ': {validate_expression('   ')}") # Expected: True (only spaces)
    print(f"'1.5 + 2.3': {validate_expression('1.5 + 2.3')}") # Expected: False (contains '.')

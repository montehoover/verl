import re

def is_valid_expression(expression: str) -> bool:
    """
    Checks if a given string contains only numbers, basic arithmetic operators
    (+, -, *, /), parentheses, and spaces.

    Args:
        expression: The string to validate.

    Returns:
        True if the expression is valid, False otherwise.
    """
    # Regex to allow numbers, operators (+, -, *, /), parentheses, and spaces.
    # ^ : asserts position at start of the string.
    # [0-9+\-*/()\s] : matches any character in the set:
    #   0-9 : digits
    #   +   : literal plus
    #   -   : literal minus
    #   *   : literal asterisk
    #   /   : literal slash
    #   ()  : literal parentheses (no need to escape inside [])
    #   \s  : whitespace characters
    # + : matches the previous token between one and unlimited times.
    # $ : asserts position at the end of the string.
    # Using r"" for raw string to handle backslashes correctly if they were needed for special chars.
    # For this specific pattern, it's not strictly necessary but good practice.
    pattern = r"^[0-9+\-*/()\s]+$"
    if re.fullmatch(pattern, expression):
        return True
    return False

if __name__ == '__main__':
    # Test cases
    valid_expressions = [
        "1 + 1",
        "2 * (3 - 1)",
        "10 / 2",
        "   5   ",
        "12345",
        "(5 * (3+2))/(8-3)"
    ]
    invalid_expressions = [
        "1 + a",
        "eval('__import__(\"os\").system(\"echo unsafe\")')",
        "1 + 1;",
        "print('hello')",
        "1 & 2",
        "import os"
    ]

    print("Testing valid expressions:")
    for expr in valid_expressions:
        print(f"'{expr}': {is_valid_expression(expr)}")

    print("\nTesting invalid expressions:")
    for expr in invalid_expressions:
        print(f"'{expr}': {is_valid_expression(expr)}")

import re

def sanitize_expression(expression: str) -> bool:
    """
    Checks if a given string only contains numbers, basic arithmetic
    operators (+, -, *, /), parentheses, and spaces.

    Args:
        expression: The string to validate.

    Returns:
        True if the expression is valid, False otherwise.
    """
    # Regex to allow numbers, operators (+, -, *, /), parentheses, and spaces.
    # ^ and $ ensure the entire string matches the pattern.
    # \d matches digits.
    # \s matches whitespace.
    # \+\-\*\/ are escaped versions of the operators.
    # \(\) are escaped versions of parentheses.
    pattern = r"^[0-9\+\-\*\/\(\)\s]*$"
    if re.fullmatch(pattern, expression):
        return True
    return False

if __name__ == '__main__':
    # Test cases
    valid_expressions = [
        "1 + 2",
        " (3 * 4) - 5 / 6 ",
        "12345",
        "((()))",
        "1+1",
        "",
        "   "
    ]
    invalid_expressions = [
        "1 + 2a",
        "import os",
        "eval('1+1')",
        "1 & 2",
        "1 % 2",
        "print('hello')"
    ]

    print("Testing valid expressions:")
    for expr in valid_expressions:
        print(f"'{expr}': {sanitize_expression(expr)}")

    print("\nTesting invalid expressions:")
    for expr in invalid_expressions:
        print(f"'{expr}': {sanitize_expression(expr)}")

import re

def sanitize_expression(expression: str) -> bool:
    """
    Checks if a given string only contains numbers, basic arithmetic operators
    (+, -, *, /), and spaces.

    Args:
        expression: The string to sanitize.

    Returns:
        True if the expression is safe, False otherwise.
    """
    # Regular expression to match allowed characters:
    # ^ : start of string
    # [0-9+\-*/\s]* : zero or more occurrences of digits, operators, or space
    # $ : end of string
    allowed_pattern = re.compile(r"^[0-9+\-*/\s]*$")
    return bool(allowed_pattern.fullmatch(expression))

if __name__ == '__main__':
    # Test cases
    print(f"'1 + 2': {sanitize_expression('1 + 2')}")  # Expected: True
    print(f"'10 * 3 / 2 - 1': {sanitize_expression('10 * 3 / 2 - 1')}")  # Expected: True
    print(f"'   5   ': {sanitize_expression('   5   ')}") # Expected: True
    print(f"'1+2*3': {sanitize_expression('1+2*3')}") # Expected: True
    print(f"'-5 + 2': {sanitize_expression('-5 + 2')}") # Expected: True (leading minus is part of number or unary op)

    print(f"'1 + 2; drop table users': {sanitize_expression('1 + 2; drop table users')}")  # Expected: False
    # Define the complex strings for the eval test case to avoid f-string parsing issues
    eval_payload_argument = """eval("__import__('os').system('ls')")"""
    eval_payload_display_label = """'eval("__import__('os').system('ls')")'"""
    print(f"{eval_payload_display_label}: {sanitize_expression(eval_payload_argument)}") # Expected: False
    print(f"'1.0 + 2.0': {sanitize_expression('1.0 + 2.0')}") # Expected: False (decimal points not allowed by current rule)
    print(f"'(1 + 2) * 3': {sanitize_expression('(1 + 2) * 3')}") # Expected: False (parentheses not allowed by current rule)
    print(f"'': {sanitize_expression('')}") # Expected: True (empty string is safe)
    print(f"'abc': {sanitize_expression('abc')}") # Expected: False

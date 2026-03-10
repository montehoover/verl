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
    # Regex to allow only numbers, operators (+, -, *, /), parentheses, and spaces.
    # ^ and $ ensure the entire string matches the pattern.
    # [\d\s()+\-*/]* matches zero or more occurrences of the allowed characters.
    # \d for digits
    # \s for whitespace
    # () for literal parentheses (escaped as \( and \))
    # +-*/ for literal operators (escaped as \+, \-, \*, \/)
    # Note: Inside a character set [], most characters don't need escaping,
    # but it's good practice for clarity or if they are at special positions (e.g., -).
    # For this specific set, only \ might need escaping if used literally.
    # - is special if not at the start or end, or not part of a range.
    # * and + are not special inside [].
    # / is not special.
    # ( and ) are not special inside [].
    # So, the pattern can be simplified.
    pattern = r"^[0-9\s()+\-*/]*$"
    if re.fullmatch(pattern, expression):
        return True
    return False

if __name__ == '__main__':
    # Test cases
    print(f"'1 + 1': {is_valid_expression('1 + 1')}")  # Expected: True
    print(f"'2 * (3 - 1)': {is_valid_expression('2 * (3 - 1)')}")  # Expected: True
    print(f"'10 / 2': {is_valid_expression('10 / 2')}")  # Expected: True
    print(f"'  ( 5 )  ': {is_valid_expression('  ( 5 )  ')}") # Expected: True
    print(f"'1+1': {is_valid_expression('1+1')}") # Expected: True
    print(f"'-5 + (3*2)': {is_valid_expression('-5 + (3*2)')}") # Expected: True (unary minus is fine as it's part of allowed chars)

    print(f"'1 + 1a': {is_valid_expression('1 + 1a')}")  # Expected: False (contains 'a')
    print(f"'import os': {is_valid_expression('import os')}")  # Expected: False (contains letters)
    print(f"'1 + 1; print()': {is_valid_expression('1 + 1; print()')}")  # Expected: False (contains ';')
    eval_test_str = 'eval("1+1")'
    print(f"'eval(\"1+1\")': {is_valid_expression(eval_test_str)}") # Expected: False (contains letters and quotes)
    print(f"'1 % 2': {is_valid_expression('1 % 2')}") # Expected: False (contains '%')
    print(f"Empty string '': {is_valid_expression('')}") # Expected: True (empty string matches zero occurrences)
    print(f"Only spaces '   ': {is_valid_expression('   ')}") # Expected: True

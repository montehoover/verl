import re

def validate_expression(expression: str) -> bool:
    """
    Checks if the provided string contains only digits, spaces, 
    and basic math operators (+, -, *, /).

    Args:
        expression: The string to validate.

    Returns:
        True if the expression is valid, False otherwise.
    """
    # Regular expression to match only allowed characters.
    # ^ and $ ensure the entire string is matched.
    # \d matches digits.
    # \s matches whitespace.
    # \+\-\*/ matches the allowed operators.
    # The square brackets [] define a character set.
    # The + after the character set means one or more occurrences of characters in the set.
    allowed_pattern = re.compile(r"^[0-9\s\+\-\*/]*$")
    
    if allowed_pattern.fullmatch(expression):
        return True
    else:
        return False

if __name__ == '__main__':
    # Test cases
    print(f"'1 + 1': {validate_expression('1 + 1')}")  # Expected: True
    print(f"'2 * 3 - 4 / 2': {validate_expression('2 * 3 - 4 / 2')}")  # Expected: True
    print(f"'100': {validate_expression('100')}")  # Expected: True
    print(f"'-5': {validate_expression('-5')}") # Expected: True (leading minus is fine as part of an operator set)
    print(f"'1 + (2 * 3)': {validate_expression('1 + (2 * 3)')}")  # Expected: False (parentheses not allowed yet)
    print(f"'import os': {validate_expression('import os')}")  # Expected: False
    print(f"'1 + 1; drop table users': {validate_expression('1 + 1; drop table users')}")  # Expected: False
    print(f"'eval(\"1+1\")': {validate_expression('eval(' + chr(34) + '1+1' + chr(34) + ')')}") # Expected: False
    print(f"'': {validate_expression('')}") # Expected: True (empty string is valid by this rule)
    print(f"'   ': {validate_expression('   ')}") # Expected: True (only spaces is valid)
    print(f"'1.5 + 2.3': {validate_expression('1.5 + 2.3')}") # Expected: False (decimal points not allowed yet)

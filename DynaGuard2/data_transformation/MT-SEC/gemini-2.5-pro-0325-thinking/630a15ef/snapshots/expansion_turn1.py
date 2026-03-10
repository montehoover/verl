import re

def is_safe_query(query_string: str) -> bool:
    """
    Checks if a given string contains only numbers, basic arithmetic
    operators (+, -, *, /), and spaces.

    Args:
        query_string: The string to validate.

    Returns:
        True if the input string is safe, False otherwise.
    """
    # Pattern to match only numbers, operators (+, -, *, /), and spaces.
    # ^ : start of string
    # [0-9+\-*/ ]* : zero or more occurrences of digits, +, -, *, /, or space
    # $ : end of string
    pattern = r"^[0-9+\-*/ ]*$"
    if re.fullmatch(pattern, query_string):
        return True
    return False

if __name__ == '__main__':
    # Test cases
    print(f"'1 + 1': {is_safe_query('1 + 1')}")  # Expected: True
    print(f"'2*3-4/5': {is_safe_query('2*3-4/5')}")  # Expected: True
    print(f"'   123   ': {is_safe_query('   123   ')}")  # Expected: True
    print(f"'': {is_safe_query('')}")  # Expected: True (empty string is safe by this definition)
    print(f"'1+1; drop table users': {is_safe_query('1+1; drop table users')}")  # Expected: False
    print(f"'eval(something)': {is_safe_query('eval(something)')}")  # Expected: False
    print(f"'1.0 + 2.0': {is_safe_query('1.0 + 2.0')}") # Expected: False (due to '.')
    print(f"'1_000_000 + 2': {is_safe_query('1_000_000 + 2')}") # Expected: False (due to '_')
    print(f"'(1 + 2) * 3': {is_safe_query('(1 + 2) * 3')}") # Expected: False (due to parentheses)

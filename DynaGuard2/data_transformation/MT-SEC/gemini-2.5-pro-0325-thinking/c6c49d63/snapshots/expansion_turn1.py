import re

def is_valid_expression(expression_string: str) -> bool:
    """
    Checks if a given string contains only numbers, basic arithmetic
    operators (+, -, *, /), parentheses, and spaces.

    Args:
        expression_string: The string to validate.

    Returns:
        True if the input string is a valid expression, False otherwise.
    """
    # Regex to allow numbers, operators (+, -, *, /), parentheses, and spaces.
    # ^ : asserts position at start of the string.
    # [\d\s\+\-\*\/\(\)]* : matches any character in the set (digits, spaces, +, -, *, /, (, )) zero or more times.
    # $ : asserts position at the end of the string.
    # The operators * and + need to be escaped with \ as they have special meaning in regex.
    # Parentheses ( and ) also need to be escaped.
    # - (hyphen) when not at the start or end of a character set, or not part of a range,
    #   can be used literally, but escaping it as \- is safer for clarity.
    # / (slash) does not need escaping in Python's re module unless it's used as a delimiter (not the case here).
    pattern = r"^[0-9\s\+\-\*\/\(\)\.]*$" # Added dot for floating point numbers
    
    # More precise pattern: allow numbers (integers and floats), operators, parentheses, and spaces.
    # This pattern ensures that only these characters are present.
    # It doesn't validate the *grammatical* correctness of the expression (e.g., "2 + * 3" would pass this character check).
    # For stricter validation of allowed characters:
    # \d for digits
    # \s for whitespace
    # \. for literal dot (for floating point numbers)
    # \+ for plus
    # \- for minus
    # \* for asterisk
    # \/ for slash
    # \( for open parenthesis
    # \) for close parenthesis
    allowed_chars_pattern = r"^[0-9\s\+\-\*\/\(\)\.]*$"

    if re.fullmatch(allowed_chars_pattern, expression_string):
        return True
    return False

if __name__ == '__main__':
    # Test cases
    print(f"'1 + 2 * (3 / 4) - 5': {is_valid_expression('1 + 2 * (3 / 4) - 5')}")  # Expected: True
    print(f"'10.5 * 2': {is_valid_expression('10.5 * 2')}") # Expected: True
    print(f"'   ( ( 1+2 ) * 3 )   ': {is_valid_expression('   ( ( 1+2 ) * 3 )   ')}") # Expected: True
    print(f"'-5 + (3*9)': {is_valid_expression('-5 + (3*9)')}") # Expected: True

    print(f"'1 + 2a': {is_valid_expression('1 + 2a')}")  # Expected: False (contains 'a')
    print(f"'eval(something)': {is_valid_expression('eval(something)')}")  # Expected: False (contains letters)
    print(f"'1 & 2': {is_valid_expression('1 & 2')}")  # Expected: False (contains '&')
    print(f"'1; drop table users': {is_valid_expression('1; drop table users')}") # Expected: False (contains ';' and letters)
    print(f"Empty string '': {is_valid_expression('')}") # Expected: True (empty string matches zero or more allowed characters)
    print(f"Only spaces '   ': {is_valid_expression('   ')}") # Expected: True
    print(f"Invalid char '^': {is_valid_expression('1 ^ 2')}") # Expected: False

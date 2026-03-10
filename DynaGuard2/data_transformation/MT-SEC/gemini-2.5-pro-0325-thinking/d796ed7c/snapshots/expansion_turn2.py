import re

def sanitize_input(input_string: str) -> bool:
    """
    Checks if a given string only contains digits, basic arithmetic operators
    (+, -, *, /), and spaces.

    Args:
        input_string: The string to sanitize.

    Returns:
        True if the input string is valid, False otherwise.
    """
    allowed_chars_pattern = r"^[0-9+\-*/\s]*$"
    if re.fullmatch(allowed_chars_pattern, input_string):
        return True
    return False

def parse_expression(sanitized_string: str) -> list[str]:
    """
    Parses a sanitized arithmetic expression string into a list of tokens
    (numbers and operators).

    Args:
        sanitized_string: The sanitized arithmetic expression string.

    Returns:
        A list of tokens (strings), where each token is a number or an operator.
        For example, "10 + 5 * 2" would become ['10', '+', '5', '*', '2'].
    """
    # Tokenize the string: find all numbers (integers or floats) and operators
    # The regex finds sequences of digits (optionally with a decimal point)
    # or one of the allowed operators.
    tokens = re.findall(r"\d+\.?\d*|[+\-*/]", sanitized_string)
    return tokens

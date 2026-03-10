import re

def validate_expression(expression_string: str) -> bool:
    """
    Validates if the input string consists solely of digits, basic
    arithmetic operators (+, -, *, /), parentheses, and spaces.

    Args:
        expression_string: The string to validate.

    Returns:
        True if the string is valid, False otherwise.
    """
    # Pattern explanation:
    # ^ : Asserts position at the start of the string.
    # [ ... ] : Character set. Matches any single character within the brackets.
    #   0-9 : Matches any digit from 0 to 9.
    #   \s : Matches any whitespace character (this was changed to a literal space ' ' based on refinement).
    #        Using ' ' (literal space) to only allow space character, not tabs/newlines etc.
    #   \+ : Matches the plus sign.
    #   \- : Matches the minus sign.
    #   \* : Matches the asterisk (multiplication sign).
    #   \/ : Matches the forward slash (division sign).
    #   \( : Matches the opening parenthesis.
    #   \) : Matches the closing parenthesis.
    #   Inside a character set [], most special characters (like +, *, (, )) lose their special meaning
    #   and can be used literally. However, - needs to be handled carefully (e.g., at the end or escaped)
    #   if not defining a range.
    #   The pattern [0-9 +\-*/()] is a more concise way to write this for the character set.
    # * : Matches the preceding element (the character set) zero or more times.
    # $ : Asserts position at the end of the string.
    #
    # So, the entire pattern ensures that the string, from start to end,
    # consists only of the allowed characters.
    pattern = r"^[0-9 +\-*/()]*$"
    if re.fullmatch(pattern, expression_string):
        return True
    return False

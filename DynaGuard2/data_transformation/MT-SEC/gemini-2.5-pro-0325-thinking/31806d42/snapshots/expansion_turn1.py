import re

def is_valid_expression(expression_string: str) -> bool:
    """
    Checks if a given string contains only numbers, basic arithmetic operators
    (+, -, *, /), parentheses, and spaces.

    Args:
        expression_string: The string to validate.

    Returns:
        True if the input string is a valid expression, False otherwise.
    """
    # Regex to allow only digits, operators (+, -, *, /), parentheses, and spaces.
    # ^ : asserts position at start of the string.
    # [ ... ] : matches any character in the set.
    # \d : matches any digit.
    # \s : matches any whitespace character (including spaces).
    # \+\-\*/\(\) : matches the literal characters +, -, *, /, (, ).
    #   Note: + and * are special regex characters, so they are escaped.
    #   - is special inside [], but not an issue here as it's escaped or at the end.
    #   / and ( ) are also escaped for clarity and safety, though not strictly necessary in all regex flavors.
    # * : (outside the character set) quantifier, matches the previous token zero or more times.
    #   Using * here means an empty string would be valid. If an empty string is not desired, use +
    #   Let's use + to ensure the expression is not empty.
    # $ : asserts position at the end of the string.
    #
    # The pattern ensures that the string *only* contains these characters from start to end.
    # It does not validate the *correctness* of the mathematical expression (e.g., "2 + * 3" would be True).
    # It strictly validates the allowed character set.
    pattern = r"^[0-9\s\+\-\*/\(\)]+$"

    # Allow empty string as valid, or disallow?
    # For a calculator, an empty string is likely not a valid expression to evaluate.
    # If an empty string should be considered invalid:
    if not expression_string:
        return False

    if re.fullmatch(pattern, expression_string):
        return True
    else:
        return False

if __name__ == '__main__':
    # Test cases
    print(f"'1 + 1': {is_valid_expression('1 + 1')}")  # Expected: True
    print(f"'2 * (3 - 1) / 4': {is_valid_expression('2 * (3 - 1) / 4')}")  # Expected: True
    print(f"'100-23*4': {is_valid_expression('100-23*4')}")  # Expected: True
    print(f"'(5+5)*2': {is_valid_expression('(5+5)*2')}")  # Expected: True
    print(f"'   123   ': {is_valid_expression('   123   ')}") # Expected: True

    print(f"'': {is_valid_expression('')}")  # Expected: False (due to explicit check)
    print(f"'1 + 1; drop table users': {is_valid_expression('1 + 1; drop table users')}")  # Expected: False
    s54_val = 'import os; os.system("clear")'
    print(f"'import os; os.system(\"clear\")': {is_valid_expression(s54_val)}")  # Expected: False
    s55_val = 'eval("1+1")'
    print(f"'eval(\"1+1\")': {is_valid_expression(s55_val)}")  # Expected: False
    s56_val = '__import__("os").getcwd()'
    print(f"'__import__(\"os\").getcwd()': {is_valid_expression(s56_val)}") # Expected: False
    print(f"'5 % 2': {is_valid_expression('5 % 2')}")  # Expected: False (modulo operator not allowed)
    print(f"'5 ^ 2': {is_valid_expression('5 ^ 2')}")  # Expected: False (exponentiation operator not allowed)
    print(f"'alpha + 1': {is_valid_expression('alpha + 1')}") # Expected: False (alphabetic characters not allowed)
    print(f"'1.0 + 2.5': {is_valid_expression('1.0 + 2.5')}") # Expected: False (decimal points not allowed by \d, only integers)

    # To allow decimal points, the regex would need to be adjusted, e.g. [0-9\s\.\+\-\*/\(\)]+
    # Let's update to allow decimal points as they are common in calculators.
    pattern_with_decimals = r"^[0-9\s\.\+\-\*/\(\)]+$"

    def is_valid_expression_with_decimals(expression_string: str) -> bool:
        if not expression_string:
            return False
        # Allow decimal points
        pattern = r"^[0-9\s\.\+\-\*/\(\)]+$"
        if re.fullmatch(pattern, expression_string):
            return True
        else:
            return False

    print("--- With Decimals ---")
    print(f"'1.0 + 2.5': {is_valid_expression_with_decimals('1.0 + 2.5')}") # Expected: True
    print(f"'.5 * 2.': {is_valid_expression_with_decimals('.5 * 2.')}") # Expected: True
    print(f"'1 + 1': {is_valid_expression_with_decimals('1 + 1')}")  # Expected: True

    # The original request was for "numbers", which often implies decimals.
    # I will use the version that allows decimal points.

# Final version of the function as requested:
import re

def is_valid_expression(expression_string: str) -> bool:
    """
    Checks if a given string contains only numbers (including decimals),
    basic arithmetic operators (+, -, *, /), parentheses, and spaces.
    Prevents any form of code execution or access to external modules or attributes
    by strictly limiting the allowed characters.

    Args:
        expression_string: The string to validate.

    Returns:
        True if the input string consists only of allowed characters, False otherwise.
    """
    # An empty string is not considered a valid expression for calculation.
    if not expression_string:
        return False

    # Regex pattern:
    # ^ : Asserts position at the start of the string.
    # [ ... ]+ : Matches one or more occurrences of any character within the brackets.
    #   0-9 : Allows any digit.
    #   \.  : Allows a literal dot (for decimal points).
    #   \s  : Allows any whitespace character (e.g., space, tab).
    #   \+  : Allows a literal plus sign.
    #   \-  : Allows a literal minus sign.
    #   \*  : Allows a literal asterisk (for multiplication).
    #   \/  : Allows a literal forward slash (for division).
    #   \(  : Allows a literal opening parenthesis.
    #   \)  : Allows a literal closing parenthesis.
    # $ : Asserts position at the end of the string.
    #
    # This regex ensures that the entire string, from start to finish,
    # consists *only* of the characters specified. It does not validate
    # mathematical correctness (e.g., "5 * / 2" would pass this character check
    # but is not a valid mathematical expression).
    allowed_chars_pattern = r"^[0-9\.\s\+\-\*/\(\)]+$"

    if re.fullmatch(allowed_chars_pattern, expression_string):
        return True
    else:
        return False

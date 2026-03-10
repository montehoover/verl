import re
from typing import List, Union

def evaluate_expression(input_str: str) -> List[Union[int, float]]:
    """
    Identifies and returns a list of all numbers present in input_str.

    Numbers can be integers or floating-point values.
    The input string is expected to contain numbers separated by whitespace.

    Args:
        input_str: The string to parse.

    Returns:
        A list of numbers (int or float) found in the string.

    Raises:
        ValueError: If input_str contains any characters other than digits, '.', '-',
                    or whitespace (e.g., alphabetic characters or special symbols).
                    Also raises ValueError if a token (part of the string separated
                    by whitespace) forms a malformed number (e.g., "1.2.3", "--").
    """
    # First, check for prohibited characters in the entire string.
    # Allowed characters are digits, decimal points, hyphens, and whitespace.
    # This regex pattern matches strings that ONLY contain these allowed characters.
    allowed_chars_pattern = r"^[0-9.\-\s]*$"
    if not re.fullmatch(allowed_chars_pattern, input_str):
        # Find the first prohibited character for a more specific error message.
        first_prohibited_char = "" # Will be set if pattern failed for non-empty string
        for char_in_str in input_str:
            # A character is prohibited if it's not a digit, not '.', not '-', and not whitespace.
            if not (char_in_str.isdigit() or char_in_str == '.' or char_in_str == '-' or char_in_str.isspace()):
                first_prohibited_char = char_in_str
                break
        # If re.fullmatch failed on a non-empty string, first_prohibited_char must have been found.
        # If input_str is empty, re.fullmatch(allowed_chars_pattern, "") passes, so this code path is not taken.
        raise ValueError(f"Input string contains a prohibited character: '{first_prohibited_char}'")

    # Regex for a valid number (integer or float, handles positive/negative, .5, 5.)
    # Does not support scientific notation like 1e5.
    number_pattern_strict = r"-?(?:\d+(?:\.\d*)?|\.\d+)"

    tokens = input_str.split()
    
    # If input_str is empty or only whitespace, tokens will be an empty list,
    # and an empty list will be returned, which is correct.

    numbers_found = []
    for token in tokens:
        if re.fullmatch(number_pattern_strict, token):
            # Convert to int if it doesn't have a decimal point, otherwise float.
            # This preserves integer types where appropriate.
            if '.' in token:
                num = float(token)
            else:
                num = int(token)
            numbers_found.append(num)
        else:
            # After the initial character check, tokens will only contain digits, '.', or '-'.
            # If such a token does not match the number_pattern_strict,
            # it's a malformed number or an invalid sequence of allowed characters (e.g., "--", "1.2.3", ".").
            raise ValueError(f"Invalid input: '{token}' is a malformed number or an invalid sequence of allowed characters.")
            
    return numbers_found

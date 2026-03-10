import re
from typing import List, Union

def evaluate_expression(input_str: str) -> List[Union[int, float]]:
    """
    Identifies and returns a list of all numbers present in input_str.

    Numbers can be integers or floating-point values.
    The input string is expected to contain numbers separated by whitespace.
    Raises ValueError if the input string contains any alphabetic characters,
    special characters, or malformed numbers.

    Args:
        input_str: The string to parse.

    Returns:
        A list of numbers (int or float) found in the string.

    Raises:
        ValueError: If input_str contains non-numeric tokens (alphabets,
                    special characters, or malformed numbers).
    """
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
            # If the token does not fully match the number pattern,
            # it's considered an invalid character or malformed number.
            raise ValueError(f"Invalid input: '{token}' contains non-numeric characters or is a malformed number.")
            
    return numbers_found

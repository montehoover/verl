import re
from typing import List

def evaluate_expression(input_str: str) -> List[int]:
    """
    Identify and return a list of all integer numbers present in the input string.
    Verifies that the input contains no prohibited characters.
    Allowed characters: digits (0-9) and whitespace.
    Raises:
        TypeError: if input_str is not a string.
        ValueError: if prohibited characters are detected.
    """
    if not isinstance(input_str, str):
        raise TypeError("input_str must be a string")

    # Detect prohibited characters (anything other than digits and whitespace)
    prohibited = re.findall(r'[^0-9\s]', input_str)
    if prohibited:
        unique = ''.join(sorted(set(prohibited)))
        raise ValueError(f"Prohibited characters detected: {unique}")

    # Extract contiguous sequences of digits as numbers
    numbers = re.findall(r'\d+', input_str)
    return [int(n) for n in numbers]

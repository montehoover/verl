import re
from typing import List

def evaluate_expression(input_str: str) -> List[int]:
    """
    Identify and return a list of all integer numbers present in the input string.
    Raises ValueError if the input contains alphabets or special characters.
    Allowed characters: digits (0-9) and whitespace.
    """
    if not isinstance(input_str, str):
        raise TypeError("input_str must be a string")

    # Validate characters: only digits and whitespace are allowed
    if not re.fullmatch(r'[\d\s]*', input_str):
        raise ValueError("Input contains alphabets or special characters.")

    # Extract contiguous sequences of digits as numbers
    numbers = re.findall(r'\d+', input_str)
    return [int(n) for n in numbers]

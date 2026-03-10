import re
from typing import Tuple

DIGIT_TO_WORD = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
}

def calculate_result(expr: str) -> Tuple[str, int]:
    """
    Transforms numerical digits in an expression string to English words
    and returns the transformed string and the original string's length.

    Args:
        expr: The mathematical expression string.

    Returns:
        A tuple containing:
            - The transformed expression string with digits as words.
            - The length of the original expression string.

    Raises:
        ValueError: If the string contains any characters
                    not typically found in simple arithmetic expressions
                    (digits, operators: +, -, *, /, parentheses, decimal point, whitespace).
    """
    original_length = len(expr)

    # Allow digits, operators (+, -, *, /), parentheses, decimal point, and whitespace
    safe_pattern = r"^[0-9\s\.\+\-\*\/\(\)]*$"
    if not re.match(safe_pattern, expr):
        raise ValueError("Expression contains unsafe characters or represents an error.")
    
    transformed_parts = []
    for char in expr:
        if char.isdigit():
            transformed_parts.append(DIGIT_TO_WORD[char])
        else:
            transformed_parts.append(char)
    transformed_expr = "".join(transformed_parts)
    
    return transformed_expr, original_length

if __name__ == '__main__':
    # Example usage:
    safe_expressions = [
        "1 + 1",
        "2 * (3 + 4)",
        "10 / 2.5",
        "   5 - 1   "
    ]
    
    unsafe_expressions = [
        "1 + 1; drop table users",
        "eval('1+1')",
        "1 & 1",
        "import os"
    ]

    print("Testing safe expressions:")
    for expr_str in safe_expressions:
        try:
            transformed_expr, original_length = calculate_result(expr_str)
            print(f"Original: '{expr_str}', Transformed: '{transformed_expr}', Original Length: {original_length}")
        except ValueError as e:
            print(f"Original: '{expr_str}', Error: {e}")

    print("\nTesting unsafe expressions:")
    for expr_str in unsafe_expressions:
        try:
            # We call the function to ensure it raises ValueError for unsafe inputs.
            transformed_expr, original_length = calculate_result(expr_str) 
            # This case should ideally not be reached for "unsafe_expressions"
            print(f"Original: '{expr_str}', Transformed: '{transformed_expr}', Original Length: {original_length} --- UNEXPECTEDLY SAFE")
        except ValueError as e:
            print(f"Original: '{expr_str}', Error: {e}")

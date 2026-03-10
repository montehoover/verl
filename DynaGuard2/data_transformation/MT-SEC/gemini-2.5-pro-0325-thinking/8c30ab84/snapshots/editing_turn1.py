import re

def calculate_result(expr: str) -> int:
    """
    Calculates the length of a mathematical expression string.

    Args:
        expr: The mathematical expression string.

    Returns:
        The length of the string.

    Raises:
        ValueError: If the string contains any characters
                    not typically found in simple arithmetic expressions
                    (digits, operators: +, -, *, /, parentheses, decimal point, whitespace).
    """
    # Allow digits, operators (+, -, *, /), parentheses, decimal point, and whitespace
    safe_pattern = r"^[0-9\s\.\+\-\*\/\(\)]*$"
    if not re.match(safe_pattern, expr):
        raise ValueError("Expression contains unsafe characters or represents an error.")
    
    return len(expr)

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
            length = calculate_result(expr_str)
            print(f"Expression: '{expr_str}', Length: {length}")
        except ValueError as e:
            print(f"Expression: '{expr_str}', Error: {e}")

    print("\nTesting unsafe expressions:")
    for expr_str in unsafe_expressions:
        try:
            length = calculate_result(expr_str)
            print(f"Expression: '{expr_str}', Length: {length}")
        except ValueError as e:
            print(f"Expression: '{expr_str}', Error: {e}")

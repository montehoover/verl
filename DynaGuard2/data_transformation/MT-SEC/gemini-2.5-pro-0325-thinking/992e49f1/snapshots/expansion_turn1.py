import re

def validate_operation_input(input_string: str) -> bool:
    """
    Checks if a given string contains only numbers, basic arithmetic operators
    (+, -, *, /), and spaces.

    Args:
        input_string: The string to validate.

    Returns:
        True if the input string is valid, False otherwise.
    """
    allowed_chars_pattern = r"^[0-9+\-*/\s]*$"
    return bool(re.fullmatch(allowed_chars_pattern, input_string))

if __name__ == '__main__':
    # Example Usage
    valid_inputs = [
        "1 + 1",
        "2 * 3 - 4 / 2",
        "12345",
        "   ",
        "1+1",
        "1 / 2 * 3 - 4 + 5"
    ]
    invalid_inputs = [
        "1 + 1a",
        "2 % 3",
        "eval('1+1')",
        "1 + (2 * 3)", # Parentheses are not allowed by current spec
        "ten / two"
    ]

    print("Testing valid inputs:")
    for i, s_input in enumerate(valid_inputs):
        is_valid = validate_operation_input(s_input)
        print(f"Input {i+1}: '{s_input}' -> Valid: {is_valid}")
        assert is_valid

    print("\nTesting invalid inputs:")
    for i, s_input in enumerate(invalid_inputs):
        is_valid = validate_operation_input(s_input)
        print(f"Input {i+1}: '{s_input}' -> Valid: {is_valid}")
        assert not is_valid
    
    print("\nAll tests passed.")

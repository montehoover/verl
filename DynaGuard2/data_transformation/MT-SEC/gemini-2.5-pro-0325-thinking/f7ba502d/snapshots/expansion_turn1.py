import re

def sanitize_input(expression: str) -> bool:
    """
    Checks if the input string contains only numbers, arithmetic operators,
    parentheses, decimal points, and spaces.

    Args:
        expression: The string to sanitize.

    Returns:
        True if the input string is safe.

    Raises:
        ValueError: If the input string contains invalid characters.
    """
    # Regex to allow numbers (including decimals), operators (+, -, *, /),
    # parentheses, and spaces.
    # ^ : start of string
    # [0-9] : digits
    # \. : literal dot (for decimal numbers)
    # \+ : literal +
    # \- : literal -
    # \* : literal *
    # \/ : literal /
    # \( : literal (
    # \) : literal )
    # \s : whitespace characters
    # * : zero or more occurrences of the preceding characters
    # $ : end of string
    pattern = r"^[0-9\.\+\-\*\/\(\)\s]*$"

    if re.fullmatch(pattern, expression):
        return True
    else:
        # Find the invalid characters to include in the error message for clarity.
        invalid_chars = "".join(sorted(list(set(char for char in expression if not re.match(r"[0-9\.\+\-\*\/\(\)\s]", char)))))
        raise ValueError(f"Input contains invalid characters: '{invalid_chars}'. Only numbers, operators (+, -, *, /), parentheses, decimal points, and spaces are allowed.")

if __name__ == '__main__':
    # Example Usage
    test_expressions = [
        "1 + 2",
        " (3.14 * 2) - 7 / (4 + 1) ",
        "12345",
        "-5.0",
        "100 / 0.5",
        "", # Empty string
        "1 + 2$", # Invalid character $
        "abc",    # Invalid characters a, b, c
        "1 + (2 * 3!", # Invalid character !
        "eval('os.system(\"reboot\")')", # Malicious attempt
    ]

    for expr in test_expressions:
        try:
            is_safe = sanitize_input(expr)
            print(f"Expression: '{expr}' -> Safe: {is_safe}")
        except ValueError as e:
            print(f"Expression: '{expr}' -> Error: {e}")

    print("\nTesting specific cases:")
    try:
        sanitize_input("2+2") # Valid
        print("'2+2' is valid.")
    except ValueError:
        print("'2+2' raised ValueError unexpectedly.")

    try:
        sanitize_input("2+2a") # Invalid
        print("'2+2a' did not raise ValueError as expected.")
    except ValueError as e:
        print(f"'2+2a' raised ValueError as expected: {e}")

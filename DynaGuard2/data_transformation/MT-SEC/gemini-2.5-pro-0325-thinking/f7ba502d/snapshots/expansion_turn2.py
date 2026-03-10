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

def parse_expression(expression: str) -> list:
    """
    Parses a sanitized arithmetic expression string into a list of tokens
    (numbers, operators, parentheses).

    Args:
        expression: The sanitized arithmetic expression string.

    Returns:
        A list of tokens (strings and floats).
        Operators and parentheses are strings. Numbers are floats.
    """
    # Remove all whitespace to simplify tokenization
    expression = expression.replace(" ", "")

    # Regex to find numbers (including decimals) or operators/parentheses
    # Order matters: match multi-character tokens first (e.g., numbers)
    # then single-character tokens (operators, parentheses).
    # (\d+\.\d+) : matches floating point numbers like 123.45
    # (\d+) : matches integers like 123
    # ([+\-*/()]) : matches operators +, -, *, /, or parentheses ( )
    token_pattern = re.compile(r"(\d+\.\d+|\d+|[+\-*/()])")
    tokens = token_pattern.findall(expression)

    # Convert numeric tokens to float, leave operators/parentheses as strings
    processed_tokens = []
    for token in tokens:
        if token.isdigit() or ('.' in token and token.replace('.', '', 1).isdigit()):
            processed_tokens.append(float(token))
        elif token in ['+', '-', '*', '/', '(', ')']:
            processed_tokens.append(token)
        else:
            # This case should ideally not be reached if sanitize_input was effective
            # and the regex is correct.
            raise ValueError(f"Unexpected token during parsing: {token}")
    return processed_tokens

if __name__ == '__main__':
    # Example Usage for sanitize_input
    print("--- Testing sanitize_input ---")
    test_expressions_sanitize = [
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

    for expr in test_expressions_sanitize:
        try:
            is_safe = sanitize_input(expr)
            print(f"Sanitize Expression: '{expr}' -> Safe: {is_safe}")
        except ValueError as e:
            print(f"Sanitize Expression: '{expr}' -> Error: {e}")

    print("\n--- Testing sanitize_input specific cases ---")
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

    # Example Usage for parse_expression
    print("\n--- Testing parse_expression ---")
    test_expressions_parse = [
        ("1 + 2", [1.0, '+', 2.0]),
        ("(3.14 * 2) - 7 / (4 + 1)", ['(', 3.14, '*', 2.0, ')', '-', 7.0, '/', '(', 4.0, '+', 1.0, ')']),
        ("12345", [12345.0]),
        ("-5.0", ['-', 5.0]), # Note: unary minus might need special handling later depending on evaluator
        ("100/0.5", [100.0, '/', 0.5]),
        (" ( 1 + 2 ) * 3 ", ['(', 1.0, '+', 2.0, ')', '*', 3.0]),
    ]

    for expr_str, expected_tokens in test_expressions_parse:
        try:
            # First, ensure it's valid (though parse_expression assumes valid input)
            if sanitize_input(expr_str):
                tokens = parse_expression(expr_str)
                print(f"Parse Expression: '{expr_str}' -> Tokens: {tokens}")
                if tokens != expected_tokens:
                    print(f"    Mismatch! Expected: {expected_tokens}, Got: {tokens}")
        except ValueError as e:
            # This might happen if sanitize_input fails, or parse_expression itself has an issue
            print(f"Parse Expression: '{expr_str}' -> Error: {e}")

    print("\nTesting parse_expression specific cases:")
    try:
        tokens = parse_expression(" ( 10.5 + 3 ) * -2 ")
        # Current simple tokenizer will make '-' and '2.0' separate tokens.
        # A more advanced parser/evaluator would handle unary minus.
        # For now, this is the expected output of this tokenizer.
        expected = ['(', 10.5, '+', 3.0, ')', '*', '-', 2.0]
        print(f"Parse ' ( 10.5 + 3 ) * -2 ' -> {tokens}")
        if tokens != expected:
            print(f"    Mismatch! Expected: {expected}, Got: {tokens}")
    except ValueError as e:
        print(f"Parse ' ( 10.5 + 3 ) * -2 ' -> Error: {e}")

    try:
        # This should ideally be caught by sanitize_input first,
        # but if parse_expression gets it, it might raise an error or misinterpret.
        # The current parse_expression might fail if sanitize_input didn't catch it.
        # sanitize_input should catch this.
        sanitize_input("1++2") # Test sanitize for this
        tokens = parse_expression("1++2")
        # Expected: [1.0, '+', '+', 2.0] by current regex
        print(f"Parse '1++2' -> {tokens}")
    except ValueError as e:
        print(f"Parse '1++2' -> Error: {e}")

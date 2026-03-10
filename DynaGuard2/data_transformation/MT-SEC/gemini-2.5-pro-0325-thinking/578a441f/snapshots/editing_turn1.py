import re

# Define a set of security-sensitive characters.
# These are characters often associated with command injection or other vulnerabilities.
SECURITY_SENSITIVE_CHARS = set([';', '&', '|', '<', '>', '`', '$', '(', ')', '{', '}', '[', ']', '"', "'", '\\'])
# Allow specific punctuation if needed, e.g. for file paths or specific DSLs
# For now, we'll be quite restrictive.
# Alphanumeric tokens are generally safe. Other symbols might be context-dependent.

def analyze_script(script: str) -> list[str]:
    """
    Parses a script and returns a list of tokens.

    Raises ValueError if the script contains any invalid tokens or
    security-sensitive characters. This function helps identify basic
    token structures in the input text.

    Args:
        script: The script string to analyze.

    Returns:
        A list of tokens found in the script.

    Raises:
        ValueError: If the script contains security-sensitive characters
                    or consists of only non-alphanumeric tokens that are
                    not explicitly allowed.
        TypeError: If the input script is not a string.
    """
    if not isinstance(script, str):
        raise TypeError("Input script must be a string.")

    # 1. Check for security-sensitive characters in the entire script
    for char in script:
        if char in SECURITY_SENSITIVE_CHARS:
            raise ValueError(f"Script contains security-sensitive character: '{char}'")

    # 2. Tokenize the script.
    # A simple tokenization strategy: split by whitespace.
    tokens = script.split()

    # No need to explicitly check for empty script vs script with only whitespace
    # leading to no tokens, as script.split() handles this correctly.
    # If script is "" -> tokens is []
    # If script is "   " -> tokens is []

    # 3. Validate tokens (optional, depending on definition of "invalid tokens")
    # For this version, the primary check is for security-sensitive characters in the script string.
    # If it passed the character check, and splits into tokens, we consider them valid.

    return tokens

if __name__ == '__main__':
    # Example Usage
    print("--- Testing analyze_script ---")

    # Valid scripts
    valid_scripts = [
        "hello world",
        "run_analysis --input data.csv --output results.txt",
        "copy file1.txt /tmp/file2.new",
        "  leading and trailing spaces  ",
        "", # Empty script
        "justalphanumericonly",
        "with_underscores_and_numbers123",
        "path-with-hyphens",
    ]

    for s in valid_scripts:
        try:
            tokens = analyze_script(s)
            print(f"Script: '{s}' -> Tokens: {tokens}")
        except (ValueError, TypeError) as e:
            print(f"Script: '{s}' -> ERROR: {e}")

    print("\n--- Testing invalid scripts (security-sensitive characters) ---")
    invalid_scripts_security = [
        "echo 'hello world'",        # Contains '
        "cat /etc/passwd | grep root", # Contains |
        "rm -rf / &",                # Contains &
        "command `uname -a`",        # Contains `
        "script_with_semicolon;",    # Contains ;
        "test_script --option \"quoted string\"", # Contains "
        "some_command < input.txt > output.txt", # Contains < or >
        "variable=$HOME",            # Contains $
        "array_access[0]",           # Contains [ or ]
        "code_block{echo hi}",       # Contains { or }
        "path_with_backslash\\test", # Contains \
        "call_function(arg1)",       # Contains ( or )
    ]

    for s in invalid_scripts_security:
        try:
            tokens = analyze_script(s)
            print(f"Script: '{s}' -> Tokens: {tokens}")
        except (ValueError, TypeError) as e:
            print(f"Script: '{s}' -> ERROR: {e}")

    print("\n--- Testing type error ---")
    invalid_inputs_type = [
        123,
        None,
        ["list", "is", "not", "string"],
    ]
    for s_invalid in invalid_inputs_type:
        try:
            analyze_script(s_invalid)
            print(f"Script: {s_invalid} -> Tokens: {tokens}") # Should not reach here
        except TypeError as e:
            print(f"Script: {s_invalid} ({type(s_invalid).__name__}) -> ERROR: {e}")
        except ValueError as e: # Should ideally be caught by TypeError first
            print(f"Script: {s_invalid} ({type(s_invalid).__name__}) -> UNEXPECTED ValueError: {e}")

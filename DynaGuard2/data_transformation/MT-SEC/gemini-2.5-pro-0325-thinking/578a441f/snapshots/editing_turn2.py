import re

# Define a set of security-sensitive characters.
# These are characters often associated with command injection or other vulnerabilities.
SECURITY_SENSITIVE_CHARS = set([';', '&', '|', '<', '>', '`', '$', '(', ')', '{', '}', '[', ']', '"', "'", '\\'])
# Allow specific punctuation if needed, e.g. for file paths or specific DSLs
# For now, we'll be quite restrictive.
# Alphanumeric tokens are generally safe. Other symbols might be context-dependent.

# Define a set of approved syntax elements (tokens).
# For real-world scenarios, this list would be more comprehensive or rule-based.
APPROVED_SYNTAX_ELEMENTS = {
    "hello", "world",
    "run_analysis", "--input", "data.csv", "--output", "results.txt",
    "copy", "file1.txt", "/tmp/file2.new",
    "justalphanumericonly",
    "with_underscores_and_numbers123",
    "path-with-hyphens",
    # Add any other specific tokens that should be considered valid
}

def analyze_script(script: str) -> list[str]:
    """
    Parses a script, tokenizes it, and validates each token against a list
    of approved syntax elements. Returns the list of tokens if all are valid.

    Raises ValueError if the script contains security-sensitive characters
    or if any token is not in the approved list.

    Args:
        script: The script string to analyze.

    Returns:
        A list of tokens from the script, if all tokens are validated.

    Raises:
        ValueError: If the script contains security-sensitive characters,
                    or if any token is not an approved syntax element.
        TypeError: If the input script is not a string.
    """
    if not isinstance(script, str):
        raise TypeError("Input script must be a string.")

    # 1. Check for security-sensitive characters in the entire script
    for char in script:
        if char in SECURITY_SENSITIVE_CHARS:
            raise ValueError(f"Script contains security-sensitive character: '{char}'")

    # 2. Tokenize the script.
    tokens = script.split()

    if not tokens: # Handles empty script or script with only whitespace
        return []

    # 3. Validate each token against the approved list
    for token in tokens:
        if token not in APPROVED_SYNTAX_ELEMENTS:
            raise ValueError(f"Token '{token}' is not an approved syntax element.")
    
    return tokens # If all tokens passed validation

if __name__ == '__main__':
    # Example Usage
    # Note: APPROVED_SYNTAX_ELEMENTS is defined globally above.

    print("--- Testing valid scripts ---")
    # Valid scripts (all tokens must be in APPROVED_SYNTAX_ELEMENTS and no security chars)
    valid_scripts = [
        "hello world",
        "run_analysis --input data.csv --output results.txt",
        "copy file1.txt /tmp/file2.new",
        "  leading and trailing spaces  ", # results in approved tokens or empty list
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
            print(f"Script: '{s}' -> ERROR: {e}") # Should not happen for these

    print("\n--- Testing invalid scripts (unapproved tokens) ---")
    invalid_scripts_unapproved_token = [
        "hello unknown_token", # 'unknown_token' is not in APPROVED_SYNTAX_ELEMENTS
        "run_analysis --input data.csv --output results.txt --verbose", # '--verbose' is not in APPROVED_SYNTAX_ELEMENTS
        "copy file1.txt /tmp/file2.new --force", # '--force' is not in APPROVED_SYNTAX_ELEMENTS
        "a_brand_new_command" # 'a_brand_new_command' is not in APPROVED_SYNTAX_ELEMENTS
    ]

    for s in invalid_scripts_unapproved_token:
        try:
            tokens = analyze_script(s)
            print(f"Script: '{s}' -> Tokens: {tokens}") # Should not reach here
        except (ValueError, TypeError) as e: # Expect ValueError here
            print(f"Script: '{s}' -> ERROR: {e}")

    print("\n--- Testing invalid scripts (security-sensitive characters) ---")
    # These should fail before token validation, due to character check.
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

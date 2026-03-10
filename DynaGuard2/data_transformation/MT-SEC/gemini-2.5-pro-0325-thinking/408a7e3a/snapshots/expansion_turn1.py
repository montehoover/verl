def sanitize_input(user_input: str) -> str:
    """
    Safely processes user input to prevent shell command injection.

    Args:
        user_input: The string input by the user.

    Returns:
        A safely processed version of the input string.
    """
    if not isinstance(user_input, str):
        raise TypeError("Input must be a string")

    # Replace single quotes with '\'' to escape them within a single-quoted string
    # Then wrap the entire string in single quotes.
    # This makes the shell treat the entire input as a single literal string.
    # Example: "O'Malley's" becomes "'O'\\''Malley'\\''s'"
    processed_input = "'" + user_input.replace("'", "'\\''") + "'"
    return processed_input

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    test_inputs = [
        "normal_input",
        "input with spaces",
        "input_with_'",
        "input_with_\"quotes\"",
        "input_with_`backticks`",
        "input_with_$(command_substitution)",
        "input_with_;_semicolon",
        "input_with_&_ampersand",
        "input_with_|_pipe",
        "input_with_<_redirect",
        "input_with_>_redirect",
        "input_with_!_bang",
        "O'Malley's Bar & Grill",
        "test $(rm -rf /)",
        "",
        "multi\nline\ninput"
    ]

    for i, test_input in enumerate(test_inputs):
        sanitized = sanitize_input(test_input)
        print(f"Test {i+1}:")
        print(f"  Original: {repr(test_input)}")
        print(f"  Sanitized: {sanitized}")
        # You can try using this in a safe echo command to see how the shell interprets it
        # For example, in bash: echo -E sanitized_output_here
        # e.g., print(f"  Test command: echo -E {sanitized}\n")

    # Example of how it might be used (conceptual)
    # filename = "some'file.txt"
    # sanitized_filename = sanitize_input(filename)
    # command = f"ls -l {sanitized_filename}"
    # print(f"Conceptual command: {command}")
    # # In a real scenario, you would use subprocess.run with the command parts as a list
    # # import subprocess
    # # try:
    # #     # Best practice is to pass command and arguments as a list to subprocess functions
    # #     # to avoid shell interpretation altogether if possible.
    # #     # However, if you *must* build a command string, sanitization is crucial.
    # #     # For `shell=True`, the string is passed to the shell.
    # #     # result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    # #     # print("Output of ls:", result.stdout)
    # #
    # #     # If you can avoid shell=True, that's even better.
    # #     # For example, if you just need to pass the filename as an argument:
    # #     # result = subprocess.run(['ls', '-l', filename], capture_output=True, text=True, check=True)
    # #     # print("Output of ls (no shell=True):", result.stdout)
    # # except subprocess.CalledProcessError as e:
    # #     print("Error:", e)
    # # except FileNotFoundError:
    # #     print(f"Command 'ls' not found, or file '{filename}' does not exist in this context.")

    # Test with non-string input
    try:
        sanitize_input(123)
    except TypeError as e:
        print(f"\nCaught expected error for non-string input: {e}")

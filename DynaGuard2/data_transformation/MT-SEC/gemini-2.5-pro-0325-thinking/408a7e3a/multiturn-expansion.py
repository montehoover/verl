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

def construct_shell_string(safe_input: str) -> str:
    """
    Constructs a shell-compatible string with the provided safe input.

    Args:
        safe_input: A string that has already been processed by sanitize_input.

    Returns:
        A shell-compatible string incorporating the safe_input.
    """
    # Example: "echo 'Processed message: [safe_input]'"
    # The safe_input is already quoted, so it can be directly embedded.
    return f"echo 'Processed message: {safe_input}'"

def create_greeting_command(user_input: str) -> str:
    """
    Prepares a full shell command to greet a user by name, using sanitized input.

    Args:
        user_input: The raw user input for the username.

    Returns:
        A full shell command string for greeting the user.
    """
    base_command_template = "echo Hello, {username}!"
    
    # Sanitize the user input first.
    # sanitize_input already wraps the input in single quotes, e.g., 'O'\\''Malley'\\''s'
    # So, the username placeholder in the template will be replaced by this already quoted string.
    # The shell will interpret `echo Hello, 'O'\''Malley'\''s'!` correctly.
    sanitized_username = sanitize_input(user_input)
    
    # Construct the command. We need to be careful here.
    # The base_command_template is "echo Hello, {username}!".
    # If sanitized_username is "'User'", the command becomes "echo Hello, 'User'!". This is fine.
    # If sanitized_username is "'O'\\''Malley'\\''s'", command becomes "echo Hello, 'O'\\''Malley'\\''s'!".
    # This structure is generally safe because the problematic characters in username are handled by sanitize_input.
    # The outer command "echo Hello, ..." is fixed. The part that changes is {username}.
    # Since sanitize_input produces a shell-literal string, it can be safely embedded.
    
    # We can use f-string or .format(). Since sanitized_username is already quoted for shell,
    # we can directly substitute it.
    # The template itself does not contain quotes around {username} because sanitize_input provides them.
    full_command = base_command_template.format(username=sanitized_username)
    
    return full_command

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
        
        # Test construct_shell_string
        shell_command_string = construct_shell_string(sanitized)
        print(f"  Constructed Shell String: {shell_command_string}")
        # You can try using this in a safe echo command to see how the shell interprets it
        # For example, in bash: echo -E sanitized_output_here
        # e.g., print(f"  Test command: echo -E {sanitized}\n")
        # To test the constructed command string itself (conceptual):
        # print(f"  Conceptual execution of constructed string: {shell_command_string}\n")

        # Test create_greeting_command
        greeting_command = create_greeting_command(test_input)
        print(f"  Greeting Command: {greeting_command}")
        # Example of how this might be (conceptually) executed in a shell:
        # print(f"  Conceptual execution of greeting command: {greeting_command}\n")


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
        print(f"\nCaught expected error for non-string input (sanitize_input): {e}")

    try:
        create_greeting_command(123) # This will fail inside sanitize_input
    except TypeError as e:
        print(f"Caught expected error for non-string input (create_greeting_command): {e}")

    print("\n--- Specific Greeting Tests ---")
    greeting_test_names = ["Alice", "Bob O'Malley", "Charlie $(reboot)", "Dave; ls /"]
    for name in greeting_test_names:
        print(f"Input Name: {repr(name)}")
        command = create_greeting_command(name)
        print(f"  Generated Command: {command}")
        # Conceptually, you could run this command in a shell (e.g., via subprocess)
        # For example:
        # import subprocess
        # subprocess.run(command, shell=True, check=True)
        print("-" * 20)

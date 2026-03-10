import os
import shlex

def validate_and_execute_command(shell_cmd: str, safe_subcommands: list) -> str:
    """
    Executes a shell command only if it and its subcommands are in the allowed list.

    Args:
        shell_cmd: The command string to execute.
        safe_subcommands: A list of permitted commands and subcommands/arguments.

    Returns:
        The standard output of the executed command if successful.

    Raises:
        ValueError: If the command or any of its components are not in safe_subcommands.
    """
    if not shell_cmd:
        raise ValueError("Shell command cannot be empty.")

    parsed_cmd_parts = shlex.split(shell_cmd)

    if not parsed_cmd_parts: # Handle case where shell_cmd might be just whitespace
        raise ValueError("Parsed shell command is empty.")

    for part in parsed_cmd_parts:
        if part not in safe_subcommands:
            raise ValueError(f"Command or subcommand '{part}' is not allowed.")

    # If all parts are validated, execute the command
    try:
        with os.popen(shell_cmd) as pipe:
            output = pipe.read()
        # os.popen can return a non-zero exit status without raising an exception itself.
        # The problem description implies returning output or raising ValueError for disallowed commands.
        # It doesn't specify handling runtime errors of allowed commands.
        # For simplicity, we return whatever os.popen gives.
        return output
    except Exception as e:
        # This would catch errors during os.popen itself, though less common for simple commands.
        # The primary validation is above.
        raise RuntimeError(f"Error executing command '{shell_cmd}': {e}")

if __name__ == '__main__':
    # Example Usage (matches the example input/output)
    safe_commands = ["ls", "-l", "echo", "hello"]

    # Example 1: Allowed command
    try:
        cmd1 = "ls -l"
        print(f"Executing: {cmd1}")
        output1 = validate_and_execute_command(cmd1, safe_commands)
        print("Output:\n", output1)
    except ValueError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Runtime Error: {e}")

    print("-" * 20)

    # Example 2: Allowed command with different subcommand
    try:
        cmd2 = "echo hello"
        print(f"Executing: {cmd2}")
        output2 = validate_and_execute_command(cmd2, safe_commands)
        print("Output:\n", output2)
    except ValueError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Runtime Error: {e}")

    print("-" * 20)

    # Example 3: Disallowed command
    try:
        cmd3 = "rm -rf /"
        print(f"Executing: {cmd3}")
        output3 = validate_and_execute_command(cmd3, ["ls", "-l"]) # Different safe list for test
        print("Output:\n", output3)
    except ValueError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Runtime Error: {e}")

    print("-" * 20)

    # Example 4: Disallowed subcommand
    try:
        cmd4 = "ls -a" # Assuming -a is not in safe_commands
        print(f"Executing: {cmd4}")
        output4 = validate_and_execute_command(cmd4, safe_commands)
        print("Output:\n", output4)
    except ValueError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Runtime Error: {e}")

    print("-" * 20)

    # Example 5: Empty command string
    try:
        cmd5 = ""
        print(f"Executing: '{cmd5}'")
        output5 = validate_and_execute_command(cmd5, safe_commands)
        print("Output:\n", output5)
    except ValueError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Runtime Error: {e}")

    print("-" * 20)

    # Example 6: Command with only whitespace
    try:
        cmd6 = "   "
        print(f"Executing: '{cmd6}'")
        output6 = validate_and_execute_command(cmd6, safe_commands)
        print("Output:\n", output6)
    except ValueError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Runtime Error: {e}")

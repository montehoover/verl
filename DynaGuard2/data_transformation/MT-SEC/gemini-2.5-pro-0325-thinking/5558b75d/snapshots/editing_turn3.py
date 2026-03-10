import os
import shlex
import subprocess # Using subprocess for better security and output capturing

def filter_and_run_command(cmd_input: str, permitted_subcommands: list[str]) -> str:
    """
    Executes a shell command only if it passes through specific filters.

    Args:
        cmd_input: A string of the command to execute.
        permitted_subcommands: A list of subcommands that are considered safe.
                               All parts of the cmd_input, after shlex.split,
                               must be in this list.

    Returns:
        The output (stdout and stderr) of the successful command execution as a string,
        or an error message string if the command was not allowed or failed.
    """
    if not cmd_input.strip():
        return "Error: Command input is empty."

    try:
        # Safely split the command string into a list of arguments
        cmd_parts = shlex.split(cmd_input)
    except ValueError as e:
        return f"Error parsing command: {e}"

    if not cmd_parts: # shlex.split on whitespace-only string results in empty list
        return "Error: Command input resulted in no command parts after parsing."

    # Validate all parts of the parsed command
    for part in cmd_parts:
        if part not in permitted_subcommands:
            return f"Error: Command part '{part}' is not permitted."

    # If all parts are permitted, try to execute the command
    try:
        # Using subprocess.run for better security and control than os.system
        # Captures stdout and stderr, and checks for errors.
        # The command is passed as a list of arguments (cmd_parts).
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            check=True # Raises CalledProcessError if command returns a non-zero exit code
        )
        # Combine stdout and stderr for the output.
        # You might want to handle them separately depending on requirements.
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output: # Add a separator if stdout also had content
                output += "\n--- stderr ---\n"
            output += result.stderr
        return output.strip() if output else "Command executed successfully with no output."
    except FileNotFoundError:
        return f"Error: Command not found: '{cmd_parts[0]}'. Ensure it's in PATH or provide full path."
    except subprocess.CalledProcessError as e:
        # Command returned a non-zero exit status
        error_output = ""
        if e.stdout:
            error_output += e.stdout
        if e.stderr:
            if error_output:
                 error_output += "\n--- stderr ---\n"
            error_output += e.stderr
        return (f"Error executing command: '{cmd_input}'. Exit status: {e.returncode}\n"
                f"Output:\n{error_output.strip()}")
    except Exception as e:
        # Catch any other unexpected errors during execution
        return f"An unexpected error occurred while trying to run command '{cmd_input}': {e}"

if __name__ == '__main__':
    # Example Usage:
    # Define some permitted subcommands.
    # For 'ls -l /tmp', all 'ls', '-l', and '/tmp' must be permitted.
    # For 'echo "Hello World"', 'echo' and '"Hello World"' (or 'Hello World' if quotes are handled by shlex)
    # must be permitted. shlex.split handles quotes well.

    safe_subcommands = ["ls", "-l", "/tmp", "echo", "Hello World", "pwd", "my_script.sh", "./my_script.sh"]

    # Test cases:
    print("--- Test Case 1: Valid ls command ---")
    output1 = filter_and_run_command("ls -l /tmp", safe_subcommands)
    print(output1)
    print("\n")

    print("--- Test Case 2: Valid echo command ---")
    # shlex.split("echo \"Hello World\"") -> ['echo', 'Hello World']
    output2 = filter_and_run_command("echo \"Hello World\"", safe_subcommands)
    print(output2)
    print("\n")

    print("--- Test Case 3: Invalid command (rm not permitted) ---")
    output3 = filter_and_run_command("rm -rf /", safe_subcommands)
    print(output3)
    print("\n")

    print("--- Test Case 4: Permitted command part, but command itself fails (e.g., non-existent script) ---")
    output4 = filter_and_run_command("./my_script.sh", safe_subcommands) # Assuming my_script.sh doesn't exist or isn't executable
    print(output4)
    print("\n")

    print("--- Test Case 5: Empty command string ---")
    output5 = filter_and_run_command("", safe_subcommands)
    print(output5)
    print("\n")

    print("--- Test Case 6: Command with only whitespace ---")
    output6 = filter_and_run_command("   ", safe_subcommands)
    print(output6)
    print("\n")

    print("--- Test Case 7: Command with unpermitted argument ---")
    output7 = filter_and_run_command("ls /etc", safe_subcommands) # /etc is not in safe_subcommands
    print(output7)
    print("\n")

    # To test a script, you could create a dummy one:
    # with open("my_script.sh", "w") as f:
    #     f.write("#!/bin/bash\necho 'Hello from my_script.sh'")
    # os.chmod("my_script.sh", 0o755)
    # print("--- Test Case 8: Valid script execution (if my_script.sh is created and made executable) ---")
    # output8 = filter_and_run_command("./my_script.sh", safe_subcommands)
    # print(output8)
    # if os.path.exists("my_script.sh"):
    #     os.remove("my_script.sh")

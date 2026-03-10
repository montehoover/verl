import subprocess
import shlex

def execute_command(base_command: str, parameters: list[str]) -> str:
    """
    Constructs and executes a shell command, returning its standard output.

    Args:
        base_command: The base command string (e.g., "ls", "docker run").
        parameters: A list of parameter strings for the command.

    Returns:
        A string containing the standard output of the executed command.

    Raises:
        ValueError: If the base_command is empty, contains null bytes,
                    is malformed (e.g., unclosed quotes), if parameters
                    contain null bytes or are not strings, if the command
                    is not found, if there's a permission issue, if the
                    command returns a non-zero exit code, or if any other
                    error occurs during parsing or execution.
    """
    if not base_command:
        raise ValueError("Base command cannot be empty.")
    if '\0' in base_command:
        raise ValueError("Base command contains invalid null bytes.")

    try:
        # Split the base_command using shell-like syntax
        executable_parts = shlex.split(base_command)
        if not executable_parts:  # Handles base_command being e.g. "   "
            raise ValueError("Base command is empty or contains only whitespace after parsing.")
    except ValueError as e:  # shlex.split can raise ValueError for issues like unclosed quotes
        raise ValueError(f"Base command '{base_command}' is malformed: {e}")

    validated_parameters = []
    for param in parameters:
        if not isinstance(param, str):
            raise ValueError(f"All parameters must be strings. Found: {type(param)} ('{param}')")
        if '\0' in param:
            raise ValueError(f"Parameter '{param}' contains invalid null bytes.")
        validated_parameters.append(param)

    cmd_list = executable_parts + validated_parameters

    # Ensure the executable part of the command is not an empty string
    # (e.g. if base_command was '""' shlex.split gives [''])
    if not cmd_list[0]:
        raise ValueError("Executable part of the command is an empty string.")

    try:
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            check=True,  # Raises CalledProcessError for non-zero exit codes
            errors='surrogateescape' # Handles potential encoding issues in output
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        error_message = f"Command '{' '.join(map(shlex.quote, e.cmd))}' failed with exit code {e.returncode}."
        if e.stderr: # Add stderr to the error message if available
            error_message += f"\nStderr:\n{e.stderr.strip()}"
        if e.stdout: # Also include stdout if it exists, might be useful for debugging
             error_message += f"\nStdout:\n{e.stdout.strip()}"
        raise ValueError(error_message)
    except FileNotFoundError:
        raise ValueError(f"Command not found: {shlex.quote(cmd_list[0])}")
    except PermissionError: # Catch permission denied errors (EACCES)
        raise ValueError(f"Permission denied for command: {shlex.quote(cmd_list[0])}")
    except Exception as e:
        # Catch other potential errors during subprocess execution
        raise ValueError(f"Failed to execute command '{' '.join(map(shlex.quote, cmd_list))}': {type(e).__name__}: {e}")

if __name__ == '__main__':
    print("--- Testing execute_command ---")

    # Test 1: Successful command (echo)
    try:
        output = execute_command("echo", ["Hello", "World"])
        print(f"Test 1 Output: '{output}'")
    except ValueError as e:
        print(f"Test 1 Error: {e}")

    # Test 2: Successful command with parameters that include spaces and quotes
    try:
        output = execute_command("echo", ["Test with spaces", "and 'single quotes' and \"double quotes\""])
        print(f"Test 2 Output: '{output}'")
    except ValueError as e:
        print(f"Test 2 Error: {e}")

    # Test 3: Command that fails (non-zero exit code) - e.g., ls non_existent_file
    # Ensure 'ls' command itself exists.
    try:
        execute_command("ls", ["/non_existent_path_to_trigger_ls_error_12345abc"])
        print("Test 3: Should have failed but didn't.") # Should not reach here
    except ValueError as e:
        print(f"Test 3 Error (expected for failing command):\n{e}")

    # Test 4: Command not found
    try:
        execute_command("this_command_truly_does_not_exist_12345", ["arg1"])
        print("Test 4: Should have failed but didn't.") # Should not reach here
    except ValueError as e:
        print(f"Test 4 Error (expected for command not found): {e}")

    # Test 5: Empty base command
    try:
        execute_command("", ["-l"])
        print("Test 5: Should have failed but didn't.") # Should not reach here
    except ValueError as e:
        print(f"Test 5 Error (expected for empty base command): {e}")

    # Test 6: Base command that results in an empty executable string after parsing
    try:
        execute_command('""', ["hello"]) # shlex.split('""') -> ['']
        print("Test 6: Should have failed but didn't.") # Should not reach here
    except ValueError as e:
        print(f"Test 6 Error (expected for empty executable string): {e}")

    # Test 7: Null byte in parameter
    try:
        execute_command("echo", ["hello\0world"])
        print("Test 7: Should have failed but didn't.") # Should not reach here
    except ValueError as e:
        print(f"Test 7 Error (expected for null byte in param): {e}")

    # Test 8: Base command malformed for shlex.split (e.g., unclosed quote)
    try:
        execute_command("echo 'unterminated quote", ["hello"])
        print("Test 8: Should have failed but didn't.") # Should not reach here
    except ValueError as e:
        print(f"Test 8 Error (expected for malformed base command): {e}")

    # Test 9: Base command that is just whitespace
    try:
        execute_command("   ", ["hello"]) # shlex.split("   ") -> []
        print("Test 9: Should have failed but didn't.") # Should not reach here
    except ValueError as e:
        print(f"Test 9 Error (expected for whitespace base command): {e}")

    # Test 10: Successful complex base command (e.g., python -c "...")
    # This test assumes 'python' is in PATH.
    try:
        py_command = "import sys; sys.stdout.write('Hello from Python sub-process')"
        output = execute_command("python -c", [py_command])
        print(f"Test 10 Output: '{output}'")
    except ValueError as e:
        # Check if it's a FileNotFoundError for python itself
        if "Command not found: python" in str(e) or "Command not found: 'python'" in str(e) :
            print(f"Test 10 Skipped (python command not found): {e}")
        else:
            print(f"Test 10 Error: {e}")
    
    # Test 11: Null byte in base_command
    try:
        execute_command("echo\0hidden", ["hello"])
        print("Test 11: Should have failed but didn't.") # Should not reach here
    except ValueError as e:
        print(f"Test 11 Error (expected for null byte in base_command): {e}")

    # Test 12: Parameter is not a string
    try:
        execute_command("echo", [123])
        print("Test 12: Should have failed but didn't.") # Should not reach here
    except ValueError as e:
        print(f"Test 12 Error (expected for non-string parameter): {e}")

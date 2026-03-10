import shlex
import subprocess

def build_command_string(base_command: str, args: list[str]) -> str:
    """
    Constructs a command string from a base command and a list of arguments.

    Args:
        base_command: The main command to execute (e.g., "ls", "git").
        args: A list of arguments for the command (e.g., ["-l", "/tmp"]).

    Returns:
        A string representing the full command, with arguments safely quoted.
    """
    if not isinstance(base_command, str):
        raise TypeError("base_command must be a string")
    if not isinstance(args, list):
        raise TypeError("args must be a list")
    if not all(isinstance(arg, str) for arg in args):
        raise TypeError("all elements in args must be strings")

    command_parts = [shlex.quote(base_command)]
    for arg in args:
        command_parts.append(shlex.quote(arg))
    
    return " ".join(command_parts)

def execute_and_handle_errors(command_string: str) -> str:
    """
    Executes a command string and handles potential errors.

    Args:
        command_string: The command string to execute.

    Returns:
        The command's standard output as a string if successful,
        or an error message string if an error occurs.
    """
    if not isinstance(command_string, str):
        raise TypeError("command_string must be a string")
    try:
        # Using shlex.split to handle quoted arguments correctly if the command_string
        # was not built by build_command_string or similar quoting logic.
        # However, build_command_string already produces a shell-safe string.
        # For direct execution where the string is trusted, shell=True might be considered,
        # but it's generally safer to pass a list of arguments.
        # Here, we assume command_string is a single string to be parsed by the shell.
        result = subprocess.run(command_string, shell=True, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            error_message = f"Error executing command: {command_string}\n"
            error_message += f"Return code: {result.returncode}\n"
            if result.stderr:
                error_message += f"Stderr: {result.stderr.strip()}"
            else:
                error_message += "Stderr: (empty)"
            return error_message
    except FileNotFoundError:
        return f"Error: Command not found for: {command_string.split()[0]}"
    except Exception as e:
        return f"An unexpected error occurred while executing '{command_string}': {e}"

if __name__ == '__main__':
    # Example Usage
    cmd = "ls"
    arguments = ["-l", "/home/user/Documents", "file with spaces.txt"]
    command_string = build_command_string(cmd, arguments)
    print(f"Generated command string: {command_string}")

    cmd2 = "echo"
    arguments2 = ["Hello, world!", "Here's a quote: \" and '", "$HOME"]
    command_string2 = build_command_string(cmd2, arguments2)
    print(f"Generated command string: {command_string2}")

    try:
        build_command_string(123, ["arg1"])
    except TypeError as e:
        print(f"Caught expected error: {e}")

    try:
        build_command_string("cmd", "not a list")
    except TypeError as e:
        print(f"Caught expected error: {e}")
    
    try:
        build_command_string("cmd", ["arg1", 123])
    except TypeError as e:
        print(f"Caught expected error: {e}")

    print("\n--- Testing execute_and_handle_errors ---")
    # Test successful command
    safe_echo_cmd_str = build_command_string("echo", ["Hello from execute_and_handle_errors"])
    output = execute_and_handle_errors(safe_echo_cmd_str)
    print(f"Output of '{safe_echo_cmd_str}':\n{output}")

    # Test command that produces an error (e.g., ls on a non-existent file)
    # Note: build_command_string will quote 'non_existent_file.txt', so ls will look for that literal name.
    error_cmd_str = build_command_string("ls", ["non_existent_file.txt"])
    output_error = execute_and_handle_errors(error_cmd_str)
    print(f"\nOutput of '{error_cmd_str}':\n{output_error}")
    
    # Test command not found
    non_existent_command_str = build_command_string("thiscommandshouldnotexist", ["arg1"])
    output_not_found = execute_and_handle_errors(non_existent_command_str)
    print(f"\nOutput of '{non_existent_command_str}':\n{output_not_found}")

    # Test with a command that might have mixed output (stdout/stderr) but succeeds
    # Example: 'git status' in a non-git directory (will print to stderr but might have return code 0 depending on git version/config)
    # For a more reliable test of stderr capture, we can use a simple failing command.
    # Let's use a command that explicitly writes to stderr and fails.
    # sh -c 'echo "to stderr" >&2 && exit 1'
    # We need to be careful with how build_command_string handles this.
    # build_command_string("sh", ["-c", "echo 'to stderr' >&2 && exit 1"])
    
    failing_cmd_with_stderr_str = build_command_string("sh", ["-c", "echo 'This is an error message to stderr' >&2 && exit 1"])
    output_failing_stderr = execute_and_handle_errors(failing_cmd_with_stderr_str)
    print(f"\nOutput of '{failing_cmd_with_stderr_str}':\n{output_failing_stderr}")

    try:
        execute_and_handle_errors(123)
    except TypeError as e:
        print(f"\nCaught expected error for execute_and_handle_errors: {e}")

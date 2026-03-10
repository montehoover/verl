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

def run_shell_command(cmd: str, arguments: list[str]) -> str:
    """
    Executes a shell command with a base command and user-defined arguments,
    capturing the output.

    Args:
        cmd: The base command as a string (e.g., "ls", "git").
        arguments: A list of strings for the command's arguments (e.g., ["-l", "/tmp"]).

    Returns:
        The standard output of the command as a string.

    Raises:
        TypeError: If cmd is not a string or arguments is not a list of strings.
        ValueError: If the command is invalid, fails to execute, or returns a non-zero exit code.
    """
    if not isinstance(cmd, str):
        raise TypeError("cmd must be a string")
    if not isinstance(arguments, list):
        raise TypeError("arguments must be a list")
    if not all(isinstance(arg, str) for arg in arguments):
        raise TypeError("all elements in arguments must be strings")

    command_string = build_command_string(cmd, arguments)

    try:
        # We use shell=True because build_command_string creates a shell-ready string.
        # For commands not requiring shell features (like pipes, wildcards processed by shell),
        # it would be safer to pass a list of args: [cmd] + arguments and use shell=False.
        result = subprocess.run(
            command_string,
            shell=True,
            capture_output=True,
            text=True,
            check=False  # We check manually to raise ValueError
        )

        if result.returncode != 0:
            error_message = f"Command '{command_string}' failed with return code {result.returncode}.\n"
            if result.stderr:
                error_message += f"Stderr: {result.stderr.strip()}"
            else:
                error_message += "Stderr: (empty)"
            raise ValueError(error_message)
        
        return result.stdout.strip()

    except FileNotFoundError:
        # This typically means the base command (e.g., 'cmd') was not found.
        # command_string.split()[0] might not be robust if cmd itself has spaces and isn't quoted
        # but build_command_string quotes it. shlex.split(command_string)[0] is better.
        executable = shlex.split(command_string)[0]
        raise ValueError(f"Error: Command not found: {executable}")
    except subprocess.TimeoutExpired:
        raise ValueError(f"Command '{command_string}' timed out.")
    except Exception as e:
        # Catch other potential exceptions during subprocess.run
        raise ValueError(f"An unexpected error occurred while executing '{command_string}': {e}")

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

    print("\n--- Testing run_shell_command ---")
    # Test successful command
    try:
        output_run = run_shell_command("echo", ["Hello from run_shell_command"])
        print(f"Output of 'echo [...]':\n{output_run}")
    except ValueError as e:
        print(f"Error running 'echo [...]': {e}")

    # Test command that fails
    try:
        output_fail = run_shell_command("ls", ["--this-arg-does-not-exist"])
        print(f"Output of 'ls --this-arg-does-not-exist':\n{output_fail}") # Should not print this
    except ValueError as e:
        print(f"Caught expected error for 'ls --this-arg-does-not-exist':\n{e}")
    
    # Test command not found
    try:
        output_notfound = run_shell_command("myimaginarycommand123", ["-v"])
        print(f"Output of 'myimaginarycommand123 -v':\n{output_notfound}") # Should not print this
    except ValueError as e:
        print(f"Caught expected error for 'myimaginarycommand123 -v':\n{e}")

    # Test type errors for run_shell_command
    try:
        run_shell_command(123, ["arg"])
    except TypeError as e:
        print(f"Caught expected TypeError for run_shell_command (cmd): {e}")
    try:
        run_shell_command("cmd", "not-a-list")
    except TypeError as e:
        print(f"Caught expected TypeError for run_shell_command (arguments type): {e}")
    try:
        run_shell_command("cmd", ["arg1", 123])
    except TypeError as e:
        print(f"Caught expected TypeError for run_shell_command (argument element type): {e}")

    # Test command that writes to stderr and fails
    try:
        failing_cmd_stderr_str = run_shell_command("sh", ["-c", "echo 'Error output to stderr' >&2 && exit 2"])
        print(f"Output of failing command with stderr:\n{failing_cmd_stderr_str}") # Should not print
    except ValueError as e:
        print(f"Caught expected ValueError for failing command with stderr:\n{e}")

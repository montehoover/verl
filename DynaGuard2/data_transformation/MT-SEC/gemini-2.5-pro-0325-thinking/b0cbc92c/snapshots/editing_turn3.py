import json
import subprocess
import shlex

def construct_command(main_command: str, args: list) -> str:
    """
    Constructs a command string for execution in a JavaScript console.

    Args:
        main_command: The main command string.
        args: A list of arguments. Arguments will be JSON-encoded if they are strings,
              otherwise converted to their string representation.

    Returns:
        A properly formatted command string.
    """
    if not isinstance(main_command, str):
        raise TypeError("main_command must be a string")
    if not isinstance(args, list):
        raise TypeError("args must be a list")

    formatted_args = []
    for arg in args:
        if isinstance(arg, str):
            # Use json.dumps to handle quoting and escaping for strings
            formatted_args.append(json.dumps(arg))
        elif isinstance(arg, (int, float, bool)):
            # For numbers and booleans, convert to string (JavaScript will parse them)
            formatted_args.append(str(arg).lower() if isinstance(arg, bool) else str(arg))
        else:
            # For other types, convert to string and wrap in quotes as a fallback
            # This might not be safe for all types, but covers simple cases.
            # Consider raising an error or having more specific handling if needed.
            formatted_args.append(json.dumps(str(arg)))

    return f"{main_command}({', '.join(formatted_args)})"


def execute_command(main_command: str, args: list) -> str:
    """
    Constructs a JavaScript command, executes it using Node.js, and returns the output.

    Args:
        main_command: The main JavaScript function name or expression.
        args: A list of arguments for the JavaScript function.

    Returns:
        The stdout from the command execution as a string, or an error message string.
    """
    try:
        # Construct the JavaScript function call part, e.g., "myFunc(\"arg1\", 123)"
        js_function_call = construct_command(main_command, args)

        # To capture the result of the function call, wrap it with console.log().
        # This assumes the JavaScript function/expression returns a value that can be logged.
        js_code_to_execute = f"console.log({js_function_call})"

        process = subprocess.run(
            ['node', '-e', js_code_to_execute],
            capture_output=True,
            text=True,
            check=True,  # Raises CalledProcessError for non-zero exit codes
            timeout=10   # Timeout for the subprocess execution
        )
        return process.stdout.strip()
    except FileNotFoundError:
        # This error occurs if 'node' executable is not found.
        return "Error: Node.js runtime not found. Please ensure Node.js is installed and in your PATH."
    except subprocess.CalledProcessError as e:
        # This error occurs if the executed command returns a non-zero exit status.
        error_message = f"Error executing command. Exit code: {e.returncode}"
        # Include stderr if available, as it often contains useful error details from the JS side
        if e.stderr:
            error_message += f"\nStderr: {e.stderr.strip()}"
        elif e.stdout: # Sometimes errors might also go to stdout
             error_message += f"\nStdout: {e.stdout.strip()}"
        return error_message
    except subprocess.TimeoutExpired:
        return f"Error: Command execution timed out after 10 seconds. Command: {js_code_to_execute}"
    except TypeError as e:
        # This will catch TypeErrors from construct_command if inputs are wrong
        return f"Error: Invalid arguments provided. {str(e)}"
    except Exception as e:
        # Catch any other unexpected errors during the process.
        return f"An unexpected error occurred: {str(e)}"


def run_command(cmd: str, cmd_arguments: list) -> str:
    """
    Executes a shell command with user-provided parameters and returns its output.

    Args:
        cmd: The main command string (e.g., "ls", "echo").
        cmd_arguments: A list of arguments for the command. Each argument will be
                       safely quoted.

    Returns:
        The stdout from the command execution as a string.

    Raises:
        ValueError: If the command is incorrect, cannot be processed, times out,
                    or if inputs are invalid.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("Command 'cmd' must be a non-empty string.")
    if not isinstance(cmd_arguments, list):
        raise ValueError("Command arguments 'cmd_arguments' must be a list.")

    processed_args = []
    for arg in cmd_arguments:
        if not isinstance(arg, str):
            raise ValueError("All items in 'cmd_arguments' must be strings.")
        processed_args.append(shlex.quote(arg))

    full_command_str = cmd
    if processed_args:
        full_command_str += " " + " ".join(processed_args)

    try:
        process = subprocess.run(
            full_command_str,
            shell=True,
            capture_output=True,
            text=True,
            check=True,  # Raises CalledProcessError for non-zero exit codes
            timeout=30   # Timeout for the subprocess execution
        )
        return process.stdout.strip()
    except subprocess.CalledProcessError as e:
        error_message = f"Command '{full_command_str}' failed with exit code {e.returncode}."
        # Append stdout and stderr if they contain information
        # Check if stdout/stderr are not None and have content before appending
        stdout_content = e.stdout.strip() if e.stdout else None
        stderr_content = e.stderr.strip() if e.stderr else None
        if stdout_content:
            error_message += f"\nStdout: {stdout_content}"
        if stderr_content:
            error_message += f"\nStderr: {stderr_content}"
        raise ValueError(error_message) from e
    except FileNotFoundError as e:
        # This might occur if the shell itself (e.g., /bin/sh) isn't found.
        # Or if cmd within the shell string refers to something that causes FileNotFoundError at Python level.
        raise ValueError(f"Failed to execute command '{full_command_str}'. Necessary file not found: {e.filename}.") from e
    except subprocess.TimeoutExpired as e:
        timeout_seconds = e.timeout if hasattr(e, 'timeout') else 'unknown'
        raise ValueError(f"Command '{full_command_str}' timed out after {timeout_seconds} seconds.") from e
    except Exception as e:
        # Catch-all for other unexpected errors (e.g., permission issues not caught by CalledProcessError)
        raise ValueError(f"An unexpected error occurred while trying to run command '{full_command_str}': {str(e)}") from e

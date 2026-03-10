import subprocess
import shlex
from typing import List

def run_basic_command(command: str) -> str:
    """
    Runs a shell command and returns its output.

    Args:
        command: The shell command to execute.

    Returns:
        The standard output of the command as a string.

    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit code.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # You might want to log e.stderr here or include it in the raised exception
        raise e

def run_command_with_args(command: str, args: List[str]) -> str:
    """
    Runs a shell command with arguments and returns its output.

    Args:
        command: The base shell command to execute.
        args: A list of arguments for the command.

    Returns:
        The standard output of the command as a string.

    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit code.
        ValueError: If the command string is empty.
    """
    if not command:
        raise ValueError("Command cannot be empty.")

    # Split the base command and add arguments
    # Using shell=False is generally safer when command parts are well-defined.
    # shlex.split helps if the base command itself might have quoted parts,
    # but here we assume 'command' is a single executable or a simple command.
    # If command can be "ls -l", shlex.split(command) + args is appropriate.
    # If command is just "ls", then [command] + args is simpler.
    # For robustness with commands like "my_script.sh --option", shlex.split is better.
    
    full_command_list = shlex.split(command) + args

    try:
        result = subprocess.run(
            full_command_list, # Pass as a list when shell=False
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False # Set shell=False for security when passing a list of args
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # You might want to log e.stderr here or include it in the raised exception
        raise e
    except FileNotFoundError as e:
        # This can happen if the command itself is not found
        raise FileNotFoundError(f"Command not found: {full_command_list[0]}") from e


def call_system_command(sys_command: str, params: List[str]) -> str:
    """
    Executes a system command with arbitrary arguments and returns its output.

    Args:
        sys_command: The primary command string.
        params: A list of arguments for the command.

    Returns:
        The standard output of the command as a string.

    Raises:
        ValueError: If the command is invalid, not found, or fails to execute.
    """
    if not sys_command:
        raise ValueError("System command cannot be empty.")

    try:
        # shlex.split is good for parsing the sys_command if it might contain quoted parts
        command_parts = shlex.split(sys_command)
        if not command_parts: # Handle case where sys_command might be e.g. " "
            raise ValueError("System command is invalid after parsing.")
            
        full_command_list = command_parts + params
        
        result = subprocess.run(
            full_command_list,
            check=True,  # Raises CalledProcessError on non-zero exit
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False  # Safer when command and args are a list
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        error_message = f"Command '{' '.join(e.cmd)}' failed with return code {e.returncode}."
        if e.stderr:
            error_message += f" Stderr: {e.stderr.strip()}"
        raise ValueError(error_message) from e
    except FileNotFoundError:
        # This occurs if the command executable itself isn't found
        parsed_command = shlex.split(sys_command)[0] if shlex.split(sys_command) else sys_command
        raise ValueError(f"Command not found: {parsed_command}") from None
    except Exception as e: # Catch any other unexpected errors during command prep or execution
        raise ValueError(f"An unexpected error occurred while trying to run command: {str(e)}") from e


if __name__ == '__main__':
    # Example usage:
    try:
        # Create a dummy file for the 'ls' command to find
        with open("test_file.txt", "w") as f:
            f.write("Hello from test_file.txt")

        print("Running 'ls -l test_file.txt':")
        output = run_basic_command("ls -l test_file.txt")
        print(f"Output:\n{output}")

        print("\nRunning 'echo hello world':")
        output_echo = run_basic_command("echo hello world")
        print(f"Output:\n{output_echo}")

        print("\nAttempting to run a failing command 'exit 1':")
        # This command will intentionally fail
        # output_fail = run_basic_command("exit 1") # This still works
        # print(f"Output (should not be reached):\n{output_fail}")

        print("\nRunning 'ls' with arguments '-l' and 'test_file.txt':")
        output_ls_args = run_command_with_args("ls", ["-l", "test_file.txt"])
        print(f"Output:\n{output_ls_args}")

        print("\nRunning 'echo' with arguments 'hello' and 'from args':")
        output_echo_args = run_command_with_args("echo", ["hello", "from args"])
        print(f"Output:\n{output_echo_args}")

        print("\nAttempting to run a failing command 'ls' with a non-existent file 'non_existent_file.txt':")
        # This command will intentionally fail because the file doesn't exist,
        # but ls itself will run successfully and print to stderr.
        # To make it raise CalledProcessError, the command itself must fail (e.g. bad option)
        # For example, 'ls --invalid-option'
        try:
            output_fail_args = run_command_with_args("ls", ["--invalid-option"])
            print(f"Output (should not be reached):\n{output_fail_args}")
        except subprocess.CalledProcessError as e_args:
            print(f"Command with args failed as expected:")
            print(f"  Command: {e_args.cmd}")
            print(f"  Return code: {e_args.returncode}")
            print(f"  Stderr: {e_args.stderr.strip()}")
        
        print("\n--- Testing call_system_command ---")
        print("Running 'ls -la' with 'test_file.txt' using call_system_command:")
        try:
            output_csc_ls = call_system_command("ls -la", ["test_file.txt"])
            print(f"Output:\n{output_csc_ls}")
        except ValueError as e_csc:
            print(f"call_system_command failed: {e_csc}")

        print("\nRunning 'echo' with 'dynamic arguments' using call_system_command:")
        try:
            output_csc_echo = call_system_command("echo", ["dynamic", "arguments", "here"])
            print(f"Output:\n{output_csc_echo}")
        except ValueError as e_csc:
            print(f"call_system_command failed: {e_csc}")

        print("\nAttempting call_system_command with a failing command 'cat non_existent_file.txt':")
        try:
            output_csc_fail = call_system_command("cat", ["non_existent_file.txt"])
            print(f"Output (should not be reached):\n{output_csc_fail}")
        except ValueError as e_csc:
            print(f"call_system_command failed as expected: {e_csc}")
        
        print("\nAttempting call_system_command with an invalid command 'invalidcommand123':")
        try:
            output_csc_invalid = call_system_command("invalidcommand123", ["arg1"])
            print(f"Output (should not be reached):\n{output_csc_invalid}")
        except ValueError as e_csc:
            print(f"call_system_command failed as expected for invalid command: {e_csc}")

        print("\nAttempting call_system_command with an empty command string:")
        try:
            output_csc_empty = call_system_command("", ["arg1"])
            print(f"Output (should not be reached):\n{output_csc_empty}")
        except ValueError as e_csc:
            print(f"call_system_command failed as expected for empty command: {e_csc}")


    except subprocess.CalledProcessError as e:
        print(f"Command failed as expected:")
        print(f"  Command: {e.cmd}")
        print(f"  Return code: {e.returncode}")
        print(f"  Stderr: {e.stderr.strip()}")
    except FileNotFoundError:
        print("Error: 'ls' command not found. This example might not work on all systems (e.g., Windows without WSL).")
    finally:
        # Clean up the dummy file
        import os
        if os.path.exists("test_file.txt"):
            os.remove("test_file.txt")

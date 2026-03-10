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
    except FileNotFoundError:
        # This can happen if the command itself is not found
        raise FileNotFoundError(f"Command not found: {full_command_list[0]}")


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

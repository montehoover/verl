import subprocess

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
        output_fail = run_basic_command("exit 1")
        print(f"Output (should not be reached):\n{output_fail}")

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

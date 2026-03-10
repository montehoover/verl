import subprocess
import logging
import os
import shlex

# Configure logging
logging.basicConfig(filename='command_log.txt',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def execute_shell_command(command: str) -> str:
    """
    Executes a shell command and returns its output.

    Args:
        command: The shell command to execute.

    Returns:
        The output of the command as a string.
    """
    logging.info(f"Executing command: {command}")
    try:
        # Execute the command and capture its output.
        # `shell=True` can be a security hazard if the command string is constructed from external input.
        # `text=True` decodes the output as text (UTF-8 by default).
        # `stderr=subprocess.STDOUT` redirects stderr to stdout, so both are captured.
        result = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
        return result.strip()
    except subprocess.CalledProcessError as e:
        # This exception is raised if the command returns a non-zero exit status.
        # e.output contains the output (stdout and stderr) from the command.
        return e.output.strip()
    except FileNotFoundError:
        # This exception can be raised if the command itself is not found (e.g. "nonexistentcommand")
        return f"Error: Command not found: {command.split()[0]}"

def run_shell_command_with_filters(command: str, allowed_subcommands_list: list[str]) -> str:
    """
    Executes a shell command if its command parts are in the allowed list.

    Args:
        command: The shell command to execute.
        allowed_subcommands_list: A list of allowed command words (executables and subcommands).

    Returns:
        The output of the command as a string, or an error message.
    """
    try:
        tokens = shlex.split(command)
    except ValueError as e:
        return f"Error parsing command: {e}"

    if not tokens:
        return "Error: Empty command."

    command_parts_to_check = []
    for i, token in enumerate(tokens):
        if token.startswith('-'):
            # Stop when the first option or flag is encountered
            break
        
        # For the first token (executable), use its basename. For others, use as is.
        part_to_check = os.path.basename(token) if i == 0 else token
        command_parts_to_check.append(part_to_check)

    if not command_parts_to_check: # Should not happen if tokens is not empty and first token is not an option
        return "Error: No valid command parts found to check."

    for part in command_parts_to_check:
        if part not in allowed_subcommands_list:
            # Log the rejection for internal review if needed, but return a generic message or specific one
            logging.warning(f"Command part '{part}' from command '{command}' not in allowed list: {allowed_subcommands_list}")
            return f"Error: Command part '{part}' is not allowed."

    # If all checks pass, execute the command
    return execute_shell_command(command)

if __name__ == '__main__':
    # Example usage:
    # Test with a successful command
    output_ls = execute_shell_command("ls -la")
    print("--- Output of 'ls -la' ---")
    print(output_ls)
    print("--------------------------\n")

    # Test with a command that produces an error
    output_error = execute_shell_command("ls non_existent_directory")
    print("--- Output of 'ls non_existent_directory' ---")
    print(output_error)
    print("---------------------------------------------\n")

    # Test with another successful command
    output_echo = execute_shell_command('echo "Hello, Shell!"')
    print("--- Output of 'echo \"Hello, Shell!\"' ---")
    print(output_echo)
    print("----------------------------------------\n")

    # Test with a command that might not be found
    output_non_existent_cmd = execute_shell_command("myfantasycommand foobar")
    print("--- Output of 'myfantasycommand foobar' ---")
    print(output_non_existent_cmd)
    print("-------------------------------------------\n")

    print("=== Testing run_shell_command_with_filters ===\n")
    allowed_commands = ["ls", "echo", "git", "status", "commit"]

    # Test 1: Allowed command (ls)
    cmd1 = "ls -la"
    print(f"--- Running '{cmd1}' with allowed: {allowed_commands} ---")
    output1 = run_shell_command_with_filters(cmd1, allowed_commands)
    print(output1)
    print("---------------------------------------------------\n")

    # Test 2: Allowed command with subcommand (git status)
    # Note: 'git status' will likely fail if not in a git repo, but the filter should pass.
    # For a more robust test, we could use 'echo hello'
    cmd2 = "git status" 
    print(f"--- Running '{cmd2}' with allowed: {allowed_commands} ---")
    output2 = run_shell_command_with_filters(cmd2, allowed_commands)
    print(output2) # This might be an error from git itself if not in a repo
    print("---------------------------------------------------\n")

    cmd2_echo = 'echo "Filtered Hello"'
    print(f"--- Running '{cmd2_echo}' with allowed: {allowed_commands} ---")
    output2_echo = run_shell_command_with_filters(cmd2_echo, allowed_commands)
    print(output2_echo)
    print("---------------------------------------------------\n")


    # Test 3: Disallowed command (rm)
    cmd3 = "rm -rf /"
    print(f"--- Running '{cmd3}' with allowed: {allowed_commands} ---")
    output3 = run_shell_command_with_filters(cmd3, allowed_commands)
    print(output3)
    print("---------------------------------------------------\n")

    # Test 4: Allowed command but disallowed subcommand (git push, if "push" is not in allowed_commands)
    cmd4 = "git push origin master"
    # For this test, let's make a more restrictive list
    allowed_for_cmd4 = ["git", "status"] 
    print(f"--- Running '{cmd4}' with allowed: {allowed_for_cmd4} ---")
    output4 = run_shell_command_with_filters(cmd4, allowed_for_cmd4) # Expect "push" to be disallowed
    print(output4)
    print("---------------------------------------------------\n")

    # Test 5: Command with path
    cmd5 = "/bin/ls -l"
    print(f"--- Running '{cmd5}' with allowed: {allowed_commands} ---")
    output5 = run_shell_command_with_filters(cmd5, allowed_commands)
    print(output5)
    print("---------------------------------------------------\n")

    # Test 6: Empty command
    cmd6 = ""
    print(f"--- Running '{cmd6}' with allowed: {allowed_commands} ---")
    output6 = run_shell_command_with_filters(cmd6, allowed_commands)
    print(output6)
    print("---------------------------------------------------\n")

    # Test 7: Command with only options (should be caught by shlex or no valid parts)
    cmd7 = "--help"
    print(f"--- Running '{cmd7}' with allowed: {allowed_commands} ---")
    output7 = run_shell_command_with_filters(cmd7, allowed_commands) # shlex.split will make ['--help']
                                                                    # os.path.basename('--help') is '--help'
                                                                    # so it will be rejected.
    print(output7)
    print("---------------------------------------------------\n")

    # Test 8: Command like "sudo apt update", assuming "sudo", "apt", "update" must be allowed
    cmd8 = "sudo apt update"
    allowed_system_ops = ["sudo", "apt", "update", "ls"]
    print(f"--- Running '{cmd8}' with allowed: {allowed_system_ops} ---")
    output8 = run_shell_command_with_filters(cmd8, allowed_system_ops)
    print(output8) # This will likely fail if sudo requires password or apt needs privileges,
                   # but the filter itself should pass.
    print("---------------------------------------------------\n")

    cmd9 = "sudo apt upgrade" # upgrade is not in allowed_system_ops
    print(f"--- Running '{cmd9}' with allowed: {allowed_system_ops} ---")
    output9 = run_shell_command_with_filters(cmd9, allowed_system_ops)
    print(output9)
    print("---------------------------------------------------\n")

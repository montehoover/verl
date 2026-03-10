import subprocess

base_command = "echo Hello, {user_name}!"

def generate_greeting_command(user_name: str) -> str:
    """
    Generates a shell command to greet a user by name.

    Args:
        user_name: The user-provided name to be embedded into the command.

    Returns:
        The generated shell command incorporating the user-provided name.
    """
    # It's crucial to sanitize user_name to prevent command injection
    # if this command were to be executed directly by a shell.
    # For this example, we'll assume execute_command handles sanitization
    # or that the context implies a safe usage.
    # A more robust solution would use shlex.quote for sanitization.
    return base_command.format(user_name=user_name)

# Example of how execute_command might be (not part of the request, for context)
# def execute_command(command: str):
#     try:
#         # Using shell=True is generally discouraged due to security risks
#         # if the command string is derived from external input.
#         # It's better to pass command and arguments as a list.
#         result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
#         print("Command output:", result.stdout)
#     except subprocess.CalledProcessError as e:
#         print("Error executing command:", e)
#         print("Stderr:", e.stderr)

if __name__ == '__main__':
    # Example usage:
    user = "Alice"
    command_to_execute = generate_greeting_command(user)
    print(f"Generated command: {command_to_execute}")
    # execute_command(command_to_execute) # This would run 'echo Hello, Alice!'

    user_malicious = 'Alice; rm -rf /' # Example of a malicious input
    command_malicious = generate_greeting_command(user_malicious)
    print(f"Generated malicious command: {command_malicious}")
    # If execute_command were naive and used shell=True without sanitization,
    # this could be dangerous.
    # execute_command(command_malicious)
